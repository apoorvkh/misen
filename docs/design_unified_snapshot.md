# Design: content-addressed snapshots and a universal worker bootstrap

Status: phases 1–3 implemented (`ProjectSnapshot`, workspace snapshot
store + job files, universal bootstrap, `env_store_dir`/`prewarm_envs`);
phases 4 (SSH/SkyPilot executors) and 5 (prune) pending. Builds on
`design_shared_env_store.md` (whose store protocol is unchanged);
supersedes the deep/shallow split.

## Motivation

Today a "snapshot" conflates three scopes with different identities and
lifetimes: the **code state** (metadata, lockfiles, built local packages —
content-addressable, shareable across submissions), the **submission**
(staged `.env` files — secret-bearing, transient), and each **job** (a
cloudpickled payload written at dispatch). It is also duplicated across two
build paths (deep: `uv sync` from the working tree at submission; shallow:
staged export + worker-side build), and its transport assumes a shared
filesystem — which planned SSH and SkyPilot executors do not have.

This design collapses all of that into one model:

- **Snapshot** = pure data: the staged code state, stored content-addressed
  *in the workspace* (`workspace.snapshots[hash] = <staged tree>`).
- **Env materialization** = one function of (snapshot, store root), run
  wherever policy dictates (submit host or worker, shared or node-local
  store).
- **Dispatch** = executor delivers (bootstrap, data-plane transport,
  snapshot ref, payload ref, env files) to a worker through whatever
  channel that executor has; the workspace's storage is the data plane
  for everything bulky.

The container analogy: snapshot ≈ image, payload ≈ job spec, env store ≈
the materialized runtime, workspace ≈ registry.

## Snapshot format and identity

One `Snapshot` class. Staged contents:

```
<snapshot>/
├── pyproject.toml, uv.lock, .python-version    # interpreter + [tool.uv] config
├── requirements.txt                            # uv export --frozen --no-emit-local
│                                               #   --no-header (deterministic bytes)
├── pixi.toml, pixi.lock                        # when the project has a conda env
└── packages/                                   # local packages, built at staging
    ├── purepkg-1.0-py3-none-any.whl            # pure python -> wheel
    └── nativepkg-2.0.tar.gz                    # native -> sdist (built on worker)
```

- **Pure wheels vs sdists.** Each local package is built with `uv build`
  (sdist + wheel). If the wheel tag is `py3-none-any`, ship the wheel
  (fast worker installs, no build deps at runtime). Otherwise ship the
  sdist: native extensions then compile *on the worker, inside the pixi
  activation*, where the locked toolchain (CC, CUDA, MKL) and the correct
  platform actually exist. This removes both the submit-platform binding
  of native wheels and any need for a pixi-wrapped build step at staging.
  If the wheel build fails at staging (no toolchain on the submit host),
  ship the sdist and warn — buildability is then verified at runtime.
  `path` dependencies that are already wheel/sdist files are copied
  verbatim.
- **Deterministic bytes.** `uv build` runs with a fixed
  `SOURCE_DATE_EPOCH` so unchanged source rebuilds byte-identical
  artifacts (uv_build/hatchling are reproducible under it; setuptools
  mostly). The requirements export is already deterministic. Result: the
  snapshot hash is stable per (code state, lock state), and resubmitting
  unchanged code republishes nothing.
- **Identity.** `hash(snapshot)` = `hash_values` over the staged tree
  (relative path + bytes per file), carried as a dedicated hash type so
  workspace stores can key it like `ResultHash`. Two-level env keying is
  unchanged: deps env by (requirements bytes, python selection, platform);
  overlay by (deps key, package bytes) — so envs share across snapshots
  that differ only in code.
- **Not in the snapshot**: env files (secrets; submission-scoped),
  payloads (job-scoped). The content-addressed store is immortal-ish and
  shareable; nothing secret or per-job belongs in it.

## Workspace: snapshot store + job blobs

Workspaces gain two facilities, mirroring the existing result store:

1. **`workspace.snapshots[hash]`** — content-addressed, immutable,
   publish-payload-before-pointer (the existing store protocol).
   - `DiskWorkspace`: a directory tree per hash under the workspace, with
     the marker/fsync mechanics of the env store.
   - `CloudWorkspace`: one tarball object per hash (avoids many-small-file
     uploads), downloaded and unpacked into the per-host local cache on
     first use — the `ObstoreResultStore` pattern.
   - `MemoryWorkspace`: tempdir-backed; local execution only.
2. **Job blobs** — small, submission/job-scoped, *not* content-addressed:
   payloads and env-file copies. Disk: files under the workspace (today's
   behavior, formalized); Cloud: objects under a `jobs/` prefix fetched by
   the worker. These are prunable by age/terminal-state, and this is where
   env files live when no shared filesystem exists (they are no more
   sensitive than the pickled payloads already stored there; the rule from
   the snapshot section is "never *content-addressed*", not "never in the
   workspace").

Payload writing moves out of `Snapshot.prepare_job` into the executor +
workspace layer. Payloads must outlive requeues (a preempted SLURM job
re-reads its payload), so retention is "until job terminal + slack", not
"until first read".

## Worker contract and bootstrap

The executor delivers five things to a worker, all strings/small files:

1. a way to run the **bootstrap** (see below),
2. the **data-plane transport** (see the transport section above; nothing
   at all for `path` transports — the paths in items 3–5 suffice),
3. **snapshot** — a directory path (`path` transport) or content key,
4. **payload** — a path or job-blob ref,
5. **env files** — paths or refs, plus the usual exec parameters (gpu
   runtime, indices, job log path).

Bootstrap phases (`misen.utils.bootstrap_env`):

1. Resolve the data plane: use argv paths directly, or fetch the
   snapshot/blobs via the transport into the env-store root
   (content-addressed, so one fetch per host per code state).
2. Ensure the **conda env** store entry first (`pixi install --frozen`).
3. Ensure the **deps env** (`uv venv` + `uv pip install -r
   requirements.txt`, hash-checked).
4. Ensure the **overlay** — installs staged wheels directly; builds sdists
   wrapped in `pixi run --frozen` so native builds see the locked
   toolchain (this ordering is why conda comes first).
5. Fetch the payload, apply activation (`VIRTUAL_ENV`, `PATH`,
   `PYTHONPATH`, pixi wrap), and `exec` `uv run --no-project --env-file …
   -m misen.utils.execute --payload …`.

**Bootstrap environment.** The bootstrap needs misen + its deps, nothing
else. Default: `uv run --with misen==<submitting version> --with
cloudpickle -m misen.utils.bootstrap_env …` — the worker needs only `uv`
on PATH (SSH/SkyPilot setup can install it) and index access. When the
submitting misen is a local/editable package (developing misen itself —
detectable from the user's `uv.lock` source table), `--with <misen wheel>`
delivered over the executor's file channel replaces the index pin; the
wheel is already built into the snapshot. Payload compatibility is *not*
a bootstrap concern: payloads are only unpickled inside the project env by
`misen.utils.execute`, where misen is the project's locked version.
LocalExecutor skips the wrapper entirely and calls the bootstrap functions
in-process (same code, no `uv run --with`, works offline).

**Env store policy subsumes deep/shallow.** Two executor knobs replace
`snapshot = true | "shallow"`:

- `env_store_dir`: where environments materialize — node-local (default
  `/tmp/misen-env-store-<user>`) or a shared path. Normalized with
  `absolute()` (never `resolve()`: symlink resolution would give one
  host's spelling of a path other hosts name differently, splitting the
  store).
- `prewarm_envs`: if true, the submit host runs the same ensure-env
  functions against `env_store_dir` at snapshot time, and jobs then
  dispatch **directly** — activation paths in argv/env, no bootstrap
  wrapper, no worker-side requirements at all (fail-fast errors,
  air-gap-friendly; the store must then be reachable by workers, and the
  workspace's job files must be worker-visible paths). Defaults:
  `LocalExecutor` prewarms (jobs run on the building host and share the
  build); `SlurmExecutor` does not (the default store is node-local, so a
  submit-host prewarm would be invisible to compute nodes — prewarm there
  requires an explicit shared `env_store_dir`).

Old deep = shared store + prewarm; old shallow = node-local store, no
prewarm. `snapshot = false` (live dispatch via `prepare_live_job`; the
in-process executor never uses snapshots) is
unchanged. The `uv sync` build path is deleted: every environment now
builds from the staged export, on whichever side policy dictates.

**The bootstrap consumes a data-plane *transport*, never a workspace.**
The bootstrap env holds only misen, so it cannot import custom
`Workspace` subclasses (and it never unpickles anything — pickles would
bind to the submitter's library versions and execute code on load).
`Workspace.bootstrap_transport()` describes the data plane in
misen-built-in terms:

- `{"kind": "path"}` — snapshot dir, payload, and env files are passed as
  plain worker-visible argv paths; the bootstrap touches no storage code
  at all. The base class derives this from `job_files_are_paths`, so
  Disk/Memory and any path-serving custom workspace get it for free.
- `{"kind": "obstore", backend, bucket, prefix, endpoint, s3_region,
  config}` — the bootstrap constructs a raw obstore client and fetches
  the snapshot tarball and job-file objects with the same module-level
  helpers `CloudWorkspace` itself uses (layouts can't drift). Fetches
  land under the env-store root: snapshots content-addressed (one fetch
  per host per code state), job files under their refs. The transport
  travels as the `MISEN_BOOTSTRAP_TRANSPORT` env override, not argv (it
  may reference credential config, and scheduler queues expose argv more
  readily than job environments).

A workspace with a data plane expressible as neither must override
`bootstrap_transport()` or be used with `prewarm_envs` (which never runs
the bootstrap). Payload *unpickling* still happens only inside the
project env, where any custom workspace's library is guaranteed present
(the user's project constructed the workspace, so it depends on its
package).

## Executors

- **LocalExecutor**: in-process bootstrap; env store local; payloads/env
  files by path. No behavioral change beyond the unified build path.
- **SlurmExecutor**: `sbatch --wrap` carries the bootstrap invocation;
  everything bulky rides the workspace. With `DiskWorkspace` this is
  today's shared-FS layout. With `CloudWorkspace` the shared-FS
  requirement disappears entirely (snapshots, payloads, env files, logs,
  results all flow through the object store — job logs already stream via
  `_LiveLogUploader`).
- **SSH executor (planned)**: `ssh` is the control channel, `scp`/sftp the
  file channel (env files with `0600`, misen wheel when needed, optionally
  the `uv` binary). Requires a workspace both sides can reach — results
  flow back only through the workspace, so plain local `DiskWorkspace`
  cannot work; `CloudWorkspace` (or a remote-mounted disk workspace) can.
  Content-addressed snapshots make repeat submissions cheap: the remote
  cache is keyed by hash, so unchanged code transfers nothing.
- **SkyPilot executor (planned)**: provisioning + `setup` installs uv;
  `file_mounts` for env files; everything else via `CloudWorkspace`, which
  is the natural (likely required) pairing.

Compatibility (validated at submit, failing early with a clear error):

| executor | Disk (shared FS) | Disk (single host) | Cloud | Memory |
|---|---|---|---|---|
| InProcess/Local | ✓ | ✓ | ✓ | ✓ |
| Slurm | ✓ | ✗ | ✓ | ✗ |
| SSH | ✓ (if mounted) | ✗ | ✓ | ✗ |
| SkyPilot | ✗ | ✗ | ✓ | ✗ |

## Lifecycle

Content-addressing makes the age-based prune (deferred since the env
store) mandatory rather than optional. One policy, several stores:
snapshot entries, env entries (deps/overlay/conda), job blobs. Reuse
already touches entry markers; dispatch additionally touches the
referenced snapshot marker so long-queued jobs hold their entries fresh.
Prune = delete entries whose marker mtime exceeds a generous TTL (weeks),
exposed as a `misen` maintenance command. Deleting a whole store remains
safe when nothing is queued or running.

## Resolved decisions

- **The requirements export stays in the snapshot** (not just `uv.lock`):
  producing it requires full project discovery — workspace-member and
  path-dep manifests at their true relative locations, which can escape
  the staged root — so it must be generated at staging time where the
  working tree exists. `uv.lock` is staged too, as provenance and as the
  input for misen-requirement resolution.
- **No shared filesystem ⇒ released misen.** A local misen checkout's
  staged wheel is only referenced by path, which requires a
  path-serving workspace; otherwise the bootstrap needs `misen==<ver>`
  from an index (presigned-URL delivery was considered and dropped).
- **LocalExecutor prewarms by default** on the node-local store: jobs run
  on the building host and can run concurrently, so building once at
  submission is strictly better than racing the first jobs into the
  store locks.
- **Sdist buildability is not verified at submission** when the local
  toolchain is missing (staging warns and ships the sdist);
  `prewarm_envs` is the fail-fast escape hatch.

## Open questions

- `uv pip install -r` behavior on a requirements file mixing hashed
  (registry) and unhashed (git/URL) entries — untested; git-dependency
  projects are the risk now that the `uv sync` path is gone.

## Phases

1. **Groundwork** — done: `SOURCE_DATE_EPOCH` in staging builds;
   pure-wheel-vs-sdist rule; conda-first materialization with
   pixi-wrapped package installs; payloads and env files moved to
   workspace job files (submission-scoped). There is no per-submission
   cleanup — payloads must outlive scheduler requeues, so all
   submission artifacts are retained for the phase-5 prune.
2. **Snapshot store** — done: `Workspace.publish/has/fetch_snapshot`
   (Disk tree + durable marker, Cloud tarball + per-host cache, Memory
   tempdir); `ProjectSnapshot` stages → hashes → publishes; executors
   reference by content key.
3. **Universal bootstrap** — done: data plane via
   `bootstrap_transport()` (plain paths, or obstore refs +
   `MISEN_BOOTSTRAP_TRANSPORT`); `uv run --no-project --with
   misen==<pin> -m misen.utils.bootstrap_env` dispatch; prewarmed
   snapshots dispatch directly; `env_store_dir` + `prewarm_envs` replace
   `snapshot="shallow"`/`env_cache`/`snapshots_dir`; the `uv sync` path
   is deleted.
4. **New executors** (pending): SSH, then SkyPilot, on the phase-3
   contract; submit-time workspace/executor compatibility validation
   beyond the current prewarm checks.
5. **Prune** (pending): age-based GC across snapshot/env/job stores +
   dispatch-time marker touches.
