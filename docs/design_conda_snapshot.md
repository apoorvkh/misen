# Design: optional conda environment in `LocalSnapshot`

## Goal

Let `LocalSnapshot` materialize an **optional conda environment** alongside the
existing uv-built venv, so users can pull in native / system libraries (CUDA
toolkit, MKL, compilers, …) without abandoning PyPI. The conda env is opt-in
via a `pixi.lock` in the project root; users without pixi get identical
behavior to today.

## Scope split

| Layer        | Owned by       | Contents                                           |
|--------------|----------------|----------------------------------------------------|
| Python + PyPI | uv venv       | interpreter, every wheel in `pyproject.toml` / `uv.lock` |
| Native / C   | conda prefix   | packages from `[dependencies]` in `pixi.toml`      |

- Python interpreter always comes from the uv venv, guaranteed by PATH
  ordering (`venv_bin:conda_bin:…` — see below). We do *not* filter
  `python` / `python_abi` / `pip` out of the conda install. If the lockfile
  pulls them in they are installed as-is: the uv venv still wins because
  `uv run` prepends its `bin` directory ahead of the conda prefix, and
  conda's activation does not set `PYTHONHOME` / `PYTHONPATH`, so
  `sys.path` can't cross-contaminate.
- `[pypi-dependencies]` in the lockfile is a hard error — PyPI belongs in
  `pyproject.toml`, not in the conda path.

## Why `pixi.lock` (not `pixi.toml` alone)

- **Reproducible by construction.** Exact versions, URLs, and hashes are
  pinned. Snapshot creation is deterministic.
- **One file signals intent.** Presence of `pixi.lock` next to CWD is the
  opt-in — no new config surface.

A user who only has `pixi.toml` runs `pixi lock` once — the same one-shot
cost they'd pay for `uv lock` in the uv world.

## Why shell out to the `pixi` CLI (not the `rattler` Python bindings)

`pixi` already knows how to install from a lockfile and activate the
resulting env. The alternative was: use `py-rattler` to install, render
an activation script, exec it in `bash`, diff `env` output. That re-
implements what `pixi` already does — with fewer conveniences and more
surface area.

Shelling out gives us:

- Correct pypi-detection, platform matching, channel priority, lockfile
  validation — all handled by pixi upstream.
- `pixi install --frozen` at snapshot time and `pixi run --frozen -x --`
  at job-spawn time are the two pixi flows we reuse; the global rattler
  package cache means installs are mostly hard-links.
- No dependency on the `py-rattler` Python binding (~33 MB).

Cost: the `pixi` binary must be on `PATH`. We document the install
(`https://pixi.sh`) and raise a targeted error if it's missing. The
population that has a `pixi.lock` already has `pixi` — it's how the
lockfile got produced.

## Project-root parameter — rejected

The user asked whether `LocalSnapshot.__init__` needs a new `project_root`
argument or can derive it. Derive it.

Rationale: the existing code already takes `Path.cwd()` as the implicit
project root — both `uv sync` (invoked without `--project`) and the env
file list (`_ENV_FILES`) resolve relative to it. Adding a parameter just
to locate `pixi.lock` would (a) break the symmetry with the env-files /
uv flows, and (b) introduce one more thing executors must thread
through. Keep the convention; look for `pixi.lock` in CWD next to where
`.env` and `uv.lock` are discovered today.

If a user needs a non-CWD root, they set CWD before creating the snapshot —
same constraint as today.

## Architecture

Everything lives in `src/misen/utils/snapshot.py`. `LocalSnapshot` gains
two attributes (`conda_manifest_path`, `pixi_bin`) and one private
method (`_snapshot_conda`). No new module-level helpers — the
pixi-specific logic is short enough to sit inline inside
`_snapshot_conda`.

```
LocalSnapshot
├── snapshot_dir                                 # parent
│   ├── python-env/                              # uv sync'd venv     (existing)
│   ├── pixi.toml, pixi.lock                     # staged manifest    (new)
│   ├── .pixi/envs/default/                      # pixi-installed env (new)
│   ├── .env, .env.local                         # copied env files   (existing)
│   ├── payloads/                                # per-job payload pickles (existing)
└── attributes
    ├── python_env_dir, env_files, payload_dir   # (existing)
    ├── conda_manifest_path: Path | None         # (new — path to staged pixi.toml; None if no pixi.lock)
    └── pixi_bin: str | None                     # (new — cached from shutil.which, same None semantics)
```

### Changes to `LocalSnapshot`

- `__init__` calls `self._snapshot_conda(self.snapshot_dir)` alongside
  `self._snapshot_python_env(...)` / `self._snapshot_env_files(...)`.
  The return value is a `Path` to the staged `pixi.toml` (what the
  pixi `--manifest-path` flag consumes), or `None` if no `pixi.lock`
  was found in CWD. `_snapshot_conda` sets `self.pixi_bin` internally
  when it needs the CLI, so `prepare_job` can use the cached binary.
- `_snapshot_conda`:
  1. Look for `pixi.lock` in CWD. If absent → return `None`.
  2. Require a `pixi.toml` adjacent to the lockfile; else raise.
  3. Line-scan the lockfile for `- pypi:` entries; raise if any.
  4. Resolve `shutil.which("pixi")`; raise a guided error if missing.
     The path is cached on `self.pixi_bin` so `prepare_job` can re-use it.
  5. Copy `pixi.toml` + `pixi.lock` into `snapshot_dir` so pixi's default
     env location (`.pixi/envs/default` relative to the manifest) lands
     under `snapshot_dir`.
  6. `pixi install --frozen --manifest-path <staged>` — install the
     env from the lockfile. `--frozen` stops pixi from updating the
     lockfile if it thinks the manifest drifted. Pre-installing means
     the first job doesn't pay install latency.
  7. Return the staged manifest path.
- `prepare_job()` builds the `uv run ...` argv as today. If
  `self.conda_manifest_path is not None` and `self.pixi_bin is not None`,
  the argv is wrapped as:
  ```
  [pixi, run, --no-progress, --color, never, --frozen,
   --manifest-path, <staged pixi.toml>, -x, --, <uv run ...>]
  ```
  - `-x` forces executable mode (no pixi-task lookup).
  - `--` stops pixi from parsing our command's flags.
  - `--frozen` ensures pixi re-uses the already-installed env without
    touching the lockfile.
  `env_overrides` stays `{"VIRTUAL_ENV": <python_env_dir>}` — conda
  activation now happens inside the pixi subprocess.
- `cleanup()`: unchanged — everything new lives under `snapshot_dir`.

### No new module-level helpers

The pixi-specific logic (lockfile discovery, pypi-deps rejection, pixi
CLI resolution) is inlined inside `_snapshot_conda`. Each piece is a few
lines and never needed outside that method, so the top level stays close
to what it was before the conda path existed.

## Environment-variable layering

Ordering (outermost → innermost, lowest → highest priority for the final
state seen by user code):

1. **Parent env**: `subprocess.Popen(env=os.environ | env_overrides)`
   seeds the child (here, `pixi run`) with the parent's `PATH`,
   `LD_LIBRARY_PATH`, etc., plus `VIRTUAL_ENV` and any executor-specific
   overrides.
2. **Conda activation by pixi**: `pixi run --frozen -x -- <cmd>`
   activates the env against live env and then execs `<cmd>`. That
   activation:
   - Sets `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `CONDA_SHLVL`.
   - Prepends `$CONDA_PREFIX/bin` to `PATH`, `$CONDA_PREFIX/lib` to
     `LD_LIBRARY_PATH` (and the macOS equivalent).
   - Sources `$CONDA_PREFIX/etc/conda/activate.d/*.sh`, where packages
     like `cudatoolkit`, `cudnn`, `mkl` inject `CUDA_HOME`,
     `CMAKE_PREFIX_PATH`, `MKL_*`, etc.
3. **uv run prepend**: the exec'd command is `uv run --no-project
   --env-file ... -m ...`. `uv run` sees `VIRTUAL_ENV` and prepends
   `<python-env>/bin` on top of the conda-activated `PATH`.
4. **User env files**: `--env-file` entries apply last and win ties
   (unchanged behavior).

Final priority visible to the child Python process:

```
PATH             = venv_bin : conda_bin : parent_path
LD_LIBRARY_PATH  = conda_lib : parent_ld_library_path
CONDA_PREFIX     = <snapshot_dir>/.pixi/envs/default
VIRTUAL_ENV      = <snapshot_dir>/python-env
CUDA_HOME, MKL_* = whatever activate.d scripts set
```

That satisfies "Python env > Conda env > System env" from the spec.

### Why re-activate per job instead of capturing env at snapshot time

An earlier iteration captured the activation env once at snapshot time
(via `pixi shell-hook --json`) and merged it into `env_overrides`. That
works and avoids one subprocess per job, but it freezes `activate.d`
logic against snapshot-time state — any script that branches on live
env (e.g. "set `CUDA_HOME` only if `CUDA_VISIBLE_DEVICES` is non-empty")
sees the wrong inputs. Re-activating per job via `pixi run` matches the
semantics of `pixi run` as users invoke it manually: activation evaluates
against the live env of the spawning process. The overhead is one extra
`pixi` process per job — a fast Rust binary with `--frozen` skipping
lockfile resolution — which is negligible compared to the Python payload
runner underneath.

## Dependency story

No new Python dependencies. `pixi` is an external CLI on `PATH`, discovered
at snapshot time via `shutil.which("pixi")`. Install instructions point
at [pixi.sh](https://pixi.sh). Users who have a `pixi.lock` already have
`pixi` (that's how the lockfile was produced).

`py-rattler` is no longer a dependency; its roles (install, activation
render, lockfile parsing) are all covered by `pixi`.

## Error modes

| Condition                                    | Behavior                                                      |
|----------------------------------------------|---------------------------------------------------------------|
| `pixi.lock` absent                           | `conda_manifest_path` / `pixi_bin` are `None` — identical behavior to today. |
| `pixi.lock` present, no adjacent `pixi.toml` | `RuntimeError`.                                               |
| `[pypi-dependencies]` present in lockfile    | `RuntimeError`: PyPI belongs in `pyproject.toml`.             |
| `pixi` CLI not on PATH                       | `RuntimeError` pointing at pixi.sh.                           |
| `pixi install` fails                         | `RuntimeError` with pixi's stderr.                            |
| `python` / `pip` in records                  | Installed as-is; uv venv wins by PATH order.                  |

## Testing

- `tests/fixtures/conda_snapshot/{pixi.toml, pixi.lock}` — tiny real
  pair for `zlib` + `xz` resolved across Linux / macOS platforms.
- `tests/test_conda_snapshot.py`: all tests drive `LocalSnapshot`
  directly — the pixi helpers are now inlined, so there's no separate
  unit layer:
  - pypi-deps rejection: synthetic lockfile with a pypi entry → raises.
  - missing-manifest rejection: lockfile without adjacent `pixi.toml`
    → raises.
  - End-to-end in a tmp CWD with both fixture files copied in, asserting:
    - `conda_manifest_path` sits under `snapshot_dir` and points to
      `pixi.toml`,
    - `pixi_bin` is populated,
    - the env exists under `snapshot_dir/.pixi/envs/default`,
    - `bin/xz` is installed and executable, `lib/libz.*` exists,
    - running the pixi-wrapped argv (the same one `prepare_job` builds)
      prints a `CONDA_PREFIX` equal to the installed prefix and a
      `PATH` whose first entry is `<prefix>/bin`,
    - `cleanup()` is a single `rmtree` and leaves no artifacts.
  - Absence case: `pixi.lock` not in CWD →
    `LocalSnapshot.conda_manifest_path is None`.

All tests skip gracefully when `pixi` is not on `PATH`
(`pytestmark = pytest.mark.skipif(...)`).

## Out of scope for this change

- Windows (pixi supports it; our PATH / `LD_LIBRARY_PATH` reasoning
  assumes POSIX).
- Multiple pixi environments (`pixi.lock` carries several named envs; we
  use the default, matching `pixi install` / `pixi shell-hook` with no
  `-e`).
- Users who keep pixi config inside `pyproject.toml` instead of a separate
  `pixi.toml` — we require `pixi.toml` explicitly. Workaround: extract a
  minimal `pixi.toml` next to `pixi.lock`.
- Caching the conda prefix across snapshots — each snapshot gets its
  own, matching the venv. The rattler global package cache amortizes
  this to hard-link cost.
