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
├── pyproject.toml, uv.lock, .python-version    # frozen uv project
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
  mostly). Result: the snapshot hash is stable per (code state, frozen uv
  project), and resubmitting unchanged code republishes nothing.
- **Identity.** `hash(snapshot)` = `hash_values` over the staged tree
  (relative path + bytes per file), carried as a dedicated hash type so
  workspace stores can key it like `ResultHash`. Two-level env keying is
  unchanged: deps env by (pyproject + lock bytes, python selection,
  platform); overlay by (deps key, package bytes) — so envs share across
  snapshots that differ only in code.
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

The executor submits one Bash bootstrap with these phases:

1. Enter through Bash and locate `uv`, installing Misen's pinned standalone
   version into the env store when no configured or PATH executable is
   available; also locate `pixi` when the staged project has a Pixi
   environment. A transport locates any additional tools it needs itself.
2. Resolve the data plane entirely from shell: use worker-visible paths
   directly, or run the workspace's Bash transport for the snapshot,
   payload, and env-file refs into local paths under the env-store root.
3. Invoke `uv run --with <locked-misen-requirement> -m misen.utils.materialize_env`
   with only those local paths. This path-only step verifies a transported
   snapshot's content key and ensures the **conda env** store entry first
   (`pixi install --frozen`).
4. Ensure the **deps env** (`uv sync --frozen --no-install-local
   --compile-bytecode` into the content-addressed entry). This consumes
   the staged uv project directly, preserving sources and explicit indexes.
5. Ensure the **overlay** — installs staged wheels directly; builds sdists
   wrapped in `pixi run --frozen` so native builds see the locked
   toolchain (this ordering is why conda comes first).
6. Apply activation (`VIRTUAL_ENV`, `PATH`, `PYTHONPATH`, pixi wrap), and
   `exec` the overlay environment's Python directly. The worker entrypoint
   applies the transported env files before loading the payload.

**Bootstrap runtime.** Bash is the root bootstrap dependency. It checks a
configured uv path and `PATH`, then downloads Misen's pinned uv version with
the official standalone installer into a versioned env-store entry. The
managed install does not alter shell profiles or self-update, and concurrent
jobs may safely publish the same pinned executable. Offline executors instead
pre-provision uv or set `MISEN_UV_BIN`; `MISEN_UV_AUTO_INSTALL=0` disables the
fallback. Submitters use the same resolver, with managed tools stored under
their XDG data directory. Pixi remains an executor-provided prerequisite.
After the shell has resolved every data-plane ref, it runs `uv run --with
<locked-misen-requirement> -m misen.utils.materialize_env …`. When the
submitting misen is a local/editable package (developing misen itself —
detectable from the user's `uv.lock` source table), its staged artifact is
usable only with a path-serving workspace. A non-path transport requires
a registry or immutable Git requirement installable on the worker. Payload
compatibility is *not* a bootstrap concern: payloads are only unpickled
inside the project env by `misen.utils.execute`, where misen is the project's
locked version.
LocalExecutor uses the same wrapper when `prewarm_envs=false`; its default
`prewarm_envs=true` path dispatches directly and needs no bootstrap.

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
  wrapper and no worker-side builds or downloads (fail-fast errors,
  air-gap-friendly; the store must then be reachable by workers, and the
  workspace's job files must be worker-visible paths). Defaults:
  `LocalExecutor` prewarms (jobs run on the building host and share the
  build); `SlurmExecutor` does not (the default store is node-local, so a
  submit-host prewarm would be invisible to compute nodes — prewarm there
  requires an explicit shared `env_store_dir`).

Old deep = shared store + prewarm; old shallow = node-local store, no
prewarm. `snapshot = false` (live dispatch via `prepare_live_job`; the
in-process executor never uses snapshots) is unchanged. Every environment
now builds through the same frozen-sync path, on whichever side policy
dictates.

**The bootstrap consumes a data-plane *transport*, never a workspace.**
The shell cannot and does not import custom `Workspace` subclasses (and
the later materializer never unpickles anything — pickles would bind to
the submitter's library versions and execute code on load).
`Workspace.bootstrap_transport()` returns Bash source for the data plane:

- `None` — snapshot dir, payload, and env files are worker-visible argv
  paths; the bootstrap touches no storage code. Path-serving workspaces
  return this explicitly.
- Bash source — Misen invokes the same script for `snapshot` and
  `job-file` operations, supplying the opaque ref and a temporary
  destination through `MISEN_TRANSPORT_*` environment variables. It also
  exposes resolved `MISEN_UV_BIN` and optional project Pixi as
  `MISEN_PIXI_BIN`. The script resolves any other tools it needs and may
  call worker CLIs or provision packages with uv or Pixi.
  The shell validates types, applies `0600` to job files, and publishes
  successful fetches into a transport-namespaced per-host cache; the
  path-only materializer verifies the snapshot content key before using
  it and evicts a corrupt tree so the next attempt can refetch it.

`CloudWorkspace` returns a small `uv run --with obstore python -c …`
script, generated by `render_python_transport()` from one ordinary static
Python function on the workspace class. The renderer verifies the fixed
function signature, rejects captured globals/closures, embeds JSON-safe
context, and adds declared PEP 508 dependencies through `uv run --with`.
This keeps the authoring and direct-test surface as normal Python while the
worker-facing contract remains Bash; raw Bash remains available to transports
that need it. Resolving the data plane therefore does not install or import
Misen or the custom workspace. The resulting transport is embedded in the
single Bash program passed to the worker, so it must not contain credentials.
Cloud authentication comes from the worker's ambient environment or workload
identity; generic `CloudWorkspace.config` values are rejected for bootstrap
dispatch rather than copied into scheduler-visible command text.

Payload *unpickling* still happens only inside the
project env, where any custom workspace's library is guaranteed present
(the user's project constructed the workspace, so it depends on its
package).

## Executors

- **LocalExecutor**: prewarmed direct dispatch by default; with prewarming
  disabled, the same Bash bootstrap and local env store as other executors.
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

- **The frozen uv project is the dependency authority.** The snapshot keeps
  the root `pyproject.toml` and `uv.lock`; `uv sync --no-install-local`
  installs only registry/Git/URL dependencies without requiring the staged
  workspace and path-dependency source trees. This preserves uv's native
  `[tool.uv.sources]` and explicit-index semantics and avoids a parallel
  requirements export.
- **No shared filesystem ⇒ remotely installable misen.** A registry lock
  becomes `misen==<ver>` and a Git lock becomes an immutable PEP 508 direct
  reference at its resolved commit. A local checkout's staged artifact is
  referenced by path and therefore still requires a path-serving workspace
  (presigned-URL delivery was considered and dropped).
- **LocalExecutor prewarms by default** on the node-local store: jobs run
  on the building host and can run concurrently, so building once at
  submission is strictly better than racing the first jobs into the
  store locks.
- **Sdist buildability is not verified at submission** when the local
  toolchain is missing (staging warns and ships the sdist);
  `prewarm_envs` is the fail-fast escape hatch.

## Phases

1. **Groundwork** — done: `SOURCE_DATE_EPOCH` in staging builds;
   pure-wheel-vs-sdist rule; conda-first materialization with
   pixi-wrapped package installs; payloads and env files moved to
   workspace job files (submission-scoped). There is no per-submission
   cleanup — payloads must outlive scheduler requeues, so all
   submission artifacts are retained for the phase-5 prune.
2. **Snapshot store** — done: `Workspace.publish_snapshot/fetch_snapshot`
   (Disk tree + durable marker, Cloud tarball + per-host cache, Memory
   tempdir); `ProjectSnapshot` stages → hashes → publishes; executors
   reference by content key.
3. **Universal bootstrap** — done: data plane via plain paths or a
   workspace Bash transport embedded in one submitted script; Bash
   resolves worker tools and data first, then path-only `uv run --no-project --with
   <locked-misen-requirement> -m misen.utils.materialize_env` materializes the
   environment and dispatches its Python directly; prewarmed
   snapshots dispatch directly; `env_store_dir` + `prewarm_envs` replace
   `snapshot="shallow"`/`env_cache`/`snapshots_dir`; dependency envs use
   the staged project's frozen uv sync on either host.
4. **New executors** (pending): SSH, then SkyPilot, on the phase-3
   contract; submit-time workspace/executor compatibility validation
   beyond the current prewarm checks.
5. **Prune** (pending): age-based GC across snapshot/env/job stores +
   dispatch-time marker touches.
