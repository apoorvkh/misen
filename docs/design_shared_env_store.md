# Design: shared env store for snapshots

> **Note:** partially superseded by `design_unified_snapshot.md`
> (implemented): environments now materialize from content-addressed
> snapshots published in the workspace, the env store root is
> `env_store_dir` (not `<snapshots>/.shared`), and `env_cache=False`
> no longer exists. The store/publication protocol, cache-dir policy,
> and lifecycle rules described below are unchanged and still
> authoritative.

## Problem

A snapshot's environment is a complete uv-built venv with the user's
packages installed non-editably into it. On a real ML project that is
gigabytes across tens of thousands of files, rebuilt for **every**
submission with pending work — any source edit changes the env. Two
compounding costs on NFS workspaces (the SUNK deployment target):

1. Every rebuild materializes the full locked dependency set, even though
   `uv.lock` changes perhaps twice per project while source changes
   constantly.
2. uv's cache defaults to the home filesystem while the workspace lives on
   the data filesystem; uv silently degrades its hardlink strategy to
   byte-for-byte copies across devices. Observed downstream: ~6.8 GB /
   32k files ≈ 2.5 minutes per submission, and one full env retained per
   code state.

## Shape of the fix

Split each environment by rate of change, and co-locate caches:

```
<snapshots_dir>/                     # default: <workspace>/tmp/snapshots
├── <random-token>/                  # per-submission snapshot (now ~MB)
│   ├── venv/                        # overlay: local packages + .pth chain
│   ├── payloads/, .env, .env.local
├── .shared/                         # shared store (token names are A-Z2-7,
│   │                                #  so ".shared" can never collide)
│   ├── python-envs/<key>/           # immutable once published
│   ├── python-envs/<key>.complete   # commit-point marker
│   ├── python-envs/<key>.lock       # NFSLock files
│   ├── uv-cache/                    # co-located UV_CACHE_DIR (policy below)
│   ├── conda-envs/<key>/            # staged pixi manifests + .pixi prefix
│   ├── conda-envs/<key>.{complete,lock}
│   └── pixi-cache/
```

- **Shared python env** (`uv sync --frozen --no-install-local
  --compile-bytecode`): all locked *remote* deps (registry/git/url), no
  local packages. Keyed by `(schema const, uv.lock bytes, .python-version
  bytes-or-absent, UV_PYTHON, sys.platform, machine)`. Interpreter
  identity is keyed by its *selection inputs* rather than a resolved
  path: an interpreter upgrade satisfying the same pin keeps the key (the
  entry stays self-consistent on its original interpreter), and if that
  interpreter is ever uninstalled, the `bin/python` reuse check — an
  `exists()` that follows symlinks — fails and the entry rebuilds.
  `--no-install-local` rather than enumerating exclusions: if
  classification ever missed a local package, exclusion-by-list would
  silently bake stale code into a shared entry; `--no-install-local`
  omits it and the gap surfaces as a loud ImportError.
  `--compile-bytecode` pre-compiles once at build time so many concurrent
  readers don't race `__pycache__` writes into the shared env over NFS.
  `uv lock` runs before keying, preserving the auto-relock semantics of
  the bare `uv sync` this replaced.
- **Per-snapshot overlay venv**: local packages only — the root project,
  workspace members, and path deps, classified from `uv.lock` `source`
  tables (`editable` / `directory` / `path`; `virtual` is never
  installed; unknown kinds raise). Built via `uv venv --python
  <shared>/bin/python` (resolves to the same base interpreter) +
  `uv pip install --no-deps --no-cache`, plus a `_misen_shared_env.pth`
  in the overlay's site-packages pointing at the shared env's
  site-packages. Jobs activate the overlay (`VIRTUAL_ENV`), so:
  - local packages shadow the shared env; runtime `pip install` lands in
    the throwaway overlay, never shared state;
  - every `sys.executable` child sees the full stack with **no** env vars
    (the `.pth` is processed by site), including children spawned with a
    scrubbed environment;
  - local entry-point scripts get real launchers in `<overlay>/bin`.
  Two env overrides complete the picture: `PATH` gains `<shared>/bin`
  (dependency console scripts like `torchrun`; `uv run` only prepends the
  overlay's bin), and `PYTHONPATH` carries the overlay site-packages as a
  safety net for children of shared-shebang scripts (e.g. `torchrun`'s
  re-exec runs the *shared* python, which never reads the overlay `.pth`).
- **Shared conda env**: same store protocol; entry = staged `pixi.toml` +
  `pixi.lock` + `.pixi/envs/default`, keyed by the two manifests'
  bytes + platform. `pixi run --frozen` activation was verified not to
  mutate the entry.

A submission with only source edits now builds: one tiny venv, a wheel per
local package, pickled payloads. Seconds, not minutes.

## Publication protocol (NFS crash-safety)

Venvs bake absolute paths into script shebangs, so the repo's usual
build-in-temp + atomic-rename publication cannot apply. Instead entries
are built **in place** at their final path and committed by a marker —
the same payload-before-pointer invariant used for result publication
(`misen.utils.task_utils.save_task_result`), with the marker as the
pointer:

1. **Fast path** (no lock): `<key>.complete` exists *and* a per-kind
   sanity file inside the entry exists → touch the marker mtime (a
   breadcrumb for future age-based pruning) and reuse.
2. **Slow path**: take `NFSLock(<key>.lock, lifetime=120, refresh=30)`.
   The workspace's usual 30/20 parameters are for millisecond holds; a
   waiter on another host reads the lockfile mtime through its NFS
   attribute cache (commonly up to 60 s stale), so a multi-minute hold
   needs lifetime − refresh headroom above that staleness or a live
   builder's lease could be broken. A blocked waiter logs who holds the
   lock (flufl records host + pid).
3. Under the lock — acquiring it wrote flufl claim files into the store
   directory, and that same-directory mutation refreshes the client's
   cached (possibly negative) dentries, so the re-checks below can't act
   on stale state and trigger a destructive recovery:
   - marker present + entry sane → reuse (built while waiting);
   - marker present, entry gutted → unlink marker, rebuild (heals the
     double-failure case: NFS server lost async writes *and* the builder
     host died before the client could resend them);
   - entry present without marker → crashed-builder residue; `rmtree`
     with retry/backoff (an orphaned `uv`/`pixi` child of a SIGKILLed
     builder may still be writing) and rebuild.
4. Build, then `os.syncfs` on the store (Linux; one syscall commits the
   mount's dirty pages — per-file fsync over 32k files would cost tens of
   seconds and defeat the point). Entry file *data* is otherwise already
   at the server via NFS close-to-open semantics when the build tool
   exits.
5. Verify `lock.is_locked()` — if the lease was stolen mid-build (extreme
   stall), a thief may already be rebuilding the entry, so publishing
   would bless a half-built directory; raise instead.
6. Publish the marker with the hash-index write mechanics: `mkstemp` in
   the store → write forensic content (host, pid, time) → fsync →
   `os.replace` → fsync the directory.

Safety of residue removal: executors dispatch jobs only after snapshot
creation returns, so an entry whose builder never published was never
referenced by any job. `NFSLock.release()` is idempotent (like
`ObjectStoreLock.release()`): a builder that lost its lease reports the
discard error, not `NotLockedError`.

## Cache-dir policy

The hardlink win exists only when cache and store share a filesystem. The
policy (identical for uv and pixi) never *prefers* the co-located cache —
it only avoids silent cross-device copy degradation:

1. Explicit `UV_CACHE_DIR` (pixi: `PIXI_CACHE_DIR` / `RATTLER_CACHE_DIR`)
   → always respected; a hint is logged if it crosses filesystems.
2. Otherwise resolve the effective cache (`uv cache dir`, which also
   honors uv config files; `pixi info --json` → `cache_dir`). Same
   `st_dev` as the store → leave it alone; it is warm (typical on
   workstations, zero duplicate downloads).
3. Different filesystem (cluster home-vs-data case) → point the build at
   `.shared/uv-cache` / `.shared/pixi-cache` and log the override with
   the env-var escape hatch. Side benefits: bulk packages live on the
   data disk instead of the (often quota'd) home FS, and deleting
   `.shared/` removes envs and cache consistently.

## Lifecycle and operations

- **No automatic GC.** The store grows one env per distinct lockfile
  state — rare by construction. Marker mtimes are touched on reuse, so an
  age-based prune can be added later. Deleting `.shared/` while no jobs
  are queued or running is always safe (everything is rebuilt on demand).
- `uv cache prune` against the co-located cache is safe: installed envs
  hold hardlinks, and inodes survive the cache-side unlink.
- **Multi-user stores** need group-writable directories *without* the
  sticky bit — breaking a dead holder's stale lock requires unlinking
  another user's lock files.
- `MemoryWorkspace` stores die with the workspace tempdir (correct, just
  cold every run). `CloudWorkspace` stores are per-host local caches — no
  new constraint, since per-snapshot dirs already had to be node-visible
  for remote executors.
- POSIX-only, matching the existing durability layer (`O_DIRECTORY`
  fsyncs). On macOS, uv links by reflink — tests assert reuse via markers
  and subprocess counts, never inode identity.
