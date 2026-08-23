"""Build the single Bash program used for non-prewarmed worker startup."""

from __future__ import annotations

import ast
import builtins
import dis
import hashlib
import inspect
import json
import shlex
from textwrap import dedent
from types import CodeType
from typing import TYPE_CHECKING

from misen.utils.uv_tool import ensure_uv_script

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

TRANSPORT_OPERATION_ENV = "MISEN_TRANSPORT_OPERATION"
TRANSPORT_REF_ENV = "MISEN_TRANSPORT_REF"
TRANSPORT_DEST_ENV = "MISEN_TRANSPORT_DEST"
PIXI_BIN_ENV = "MISEN_PIXI_BIN"


def _quote(value: str | Path) -> str:
    """Quote one value for literal assignment in generated Bash."""
    return shlex.quote(str(value))


def _array(values: list[str]) -> str:
    """Render values as a safely quoted Bash array literal."""
    return " ".join(_quote(value) for value in values)


def _shell_block(source: str) -> str:
    """Normalize one indented multiline shell block."""
    return dedent(source).strip()


def _extract_transport_function(function: Callable[..., None]) -> tuple[str, str]:
    """Return the name and standalone source of a worker transport function."""
    function = inspect.unwrap(function)
    if (
        not inspect.isfunction(function)
        or inspect.iscoroutinefunction(function)
        or inspect.isgeneratorfunction(function)
    ):
        msg = "Python transports must be synchronous, source-backed Python functions."
        raise TypeError(msg)

    parameters = inspect.signature(function).parameters
    if tuple(parameters) != ("context", "operation", "ref", "destination") or any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameter.default is not inspect.Parameter.empty
        for parameter in parameters.values()
    ):
        msg = "Python transports must have the signature (context, operation, ref, destination)."
        raise TypeError(msg)

    def global_loads(code: CodeType) -> set[str]:
        names = {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
            and isinstance(instruction.argval, str)
            and not hasattr(builtins, instruction.argval)
        }
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                names.update(global_loads(constant))
        return names

    captured = sorted({*function.__code__.co_freevars, *global_loads(function.__code__)})
    if captured:
        msg = (
            "Python transports cannot capture closures or module globals; import dependencies inside the function "
            f"and pass workspace state through context (captured: {', '.join(captured)})."
        )
        raise ValueError(msg)

    try:
        source = dedent(inspect.getsource(function))
    except (OSError, TypeError) as e:
        msg = f"Could not read source for Python transport {function.__qualname__!r}."
        raise ValueError(msg) from e
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(definition := tree.body[0], ast.FunctionDef):
        msg = f"Python transport {function.__qualname__!r} must contain exactly one function definition."
        raise ValueError(msg)
    definition.decorator_list = []
    return definition.name, ast.unparse(ast.fix_missing_locations(definition))


def render_python_transport(
    function: Callable[..., None],
    *,
    requirements: Sequence[str] = (),
    context: Mapping[str, object],
) -> str:
    """Render a source-backed Python function as a Bash workspace transport.

    The function is extracted on the submitter and invoked on the worker as
    ``function(context, operation, ref, destination)``. It must import every
    dependency it uses inside its body and may not capture globals or closures.
    ``requirements`` are PEP 508 strings installed by ``uv run --with``.

    Context is embedded in scheduler-visible shell text. It must therefore be
    JSON-safe, stable for the workspace identity, and free of credentials.

    Raises:
        TypeError: If the function or requirements do not match the contract.
        ValueError: If source extraction or context serialization fails.
    """
    name, source = _extract_transport_function(function)
    if isinstance(requirements, str) or any(not isinstance(item, str) or not item.strip() for item in requirements):
        msg = "Python transport requirements must be non-empty PEP 508 strings."
        raise TypeError(msg)
    try:
        encoded_context = json.dumps(dict(context), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as e:
        msg = "Python transport context must be JSON-serializable."
        raise ValueError(msg) from e

    program = "\n\n".join(
        (
            _shell_block(
                """
                from __future__ import annotations

                import json as __misen_json
                import os as __misen_os
                from pathlib import Path as __MisenPath
                """
            ),
            source,
            _shell_block(
                f"""
                {name}(
                    __misen_json.loads({encoded_context!r}),
                    __misen_os.environ[{TRANSPORT_OPERATION_ENV!r}],
                    __misen_os.environ[{TRANSPORT_REF_ENV!r}],
                    __MisenPath(__misen_os.environ[{TRANSPORT_DEST_ENV!r}]),
                )
                """
            ),
        )
    )
    compile(program, f"<{name}-transport>", "exec")
    dependency_args = _array([value for requirement in requirements for value in ("--with", requirement)])
    return _shell_block(
        f"""
        python_transport=(
            "$MISEN_UV_BIN" run --no-project
            {dependency_args}
            python -c {_quote(program)}
        )
        exec "${{python_transport[@]}}"
        """
    )


def worker_bootstrap_script(
    *,
    uv_bin: str,
    pixi_bin: str | None,
    requires_pixi: bool,
    transport_script: str | None,
    misen_requirement: str,
    python_version: str,
    store_root: Path,
    project_dir: Path | None,
    snapshot_key: str | None,
    payload: str,
    env_files: list[str],
    worker_args: list[str],
) -> str:
    """Return the complete Bash bootstrap submitted to a worker.

    The generated program resolves tools, fetches non-path data by executing
    the workspace's shell transport, and then invokes the path-only
    environment materializer.

    Raises:
        ValueError: If path/ref arguments do not match the transport mode.
    """
    if transport_script is None:
        if project_dir is None or snapshot_key is not None:
            msg = "Path bootstrap requires project_dir and no snapshot_key."
            raise ValueError(msg)
    elif project_dir is not None or snapshot_key is None:
        msg = "Transport bootstrap requires snapshot_key and no project_dir."
        raise ValueError(msg)

    blocks = [
        _shell_block(
            f"""
            set -euo pipefail

            store_root={_quote(store_root)}
            mkdir -p -- "$store_root"

            {ensure_uv_script(uv_bin, store_root)}

            resolve_tool() {{
                local preferred="$1"
                local name="$2"
                local resolved

                if [[ -n "$preferred" && -x "$preferred" ]]; then
                    printf '%s\\n' "$preferred"
                elif resolved="$(command -v "$name")"; then
                    printf '%s\\n' "$resolved"
                else
                    printf 'misen bootstrap: required tool `%s` is not available on this worker\\n' "$name" >&2
                    return 127
                fi
            }}
            misen_requirement={_quote(misen_requirement)}
            """
        )
    ]

    if requires_pixi:
        blocks.append(f'MISEN_PIXI_BIN="$(resolve_tool {_quote(pixi_bin or "")} pixi)"\nexport MISEN_PIXI_BIN')

    if transport_script is None:
        blocks.append(
            f"project_dir={_quote(project_dir or '')}\n"
            f"payload_path={_quote(payload)}\n"
            f"env_file_paths=({_array(env_files)})"
        )
    else:
        transport_key = hashlib.sha256(transport_script.encode()).hexdigest()
        payload_key = hashlib.sha256(payload.encode()).hexdigest()
        env_file_keys = [hashlib.sha256(ref.encode()).hexdigest() for ref in env_files]
        blocks.extend(
            (
                f"transport_script={_quote(transport_script)}",
                _shell_block(
                    r"""
                    run_transport() {
                        local operation="$1"
                        local ref="$2"
                        local destination="$3"

                        MISEN_TRANSPORT_OPERATION="$operation" \
                        MISEN_TRANSPORT_REF="$ref" \
                        MISEN_TRANSPORT_DEST="$destination" \
                            "$BASH" -euo pipefail -c "$transport_script"
                    }

                    fetch_transport() {
                        local operation="$1"
                        local ref="$2"
                        local target="$3"
                        local kind="$4"
                        if [[ "$kind" == directory && -d "$target" ]] || [[ "$kind" == file && -f "$target" ]]; then
                            return 0
                        fi

                        mkdir -p -- "$(dirname -- "$target")"
                        local temp_root
                        temp_root="$(mktemp -d "${target}.tmp.XXXXXX")"
                        local temp_target="$temp_root/$operation"

                        if ! run_transport "$operation" "$ref" "$temp_target"; then
                            rm -rf -- "$temp_root"
                            return 1
                        fi

                        if [[ "$kind" == directory ]]; then
                            if [[ ! -d "$temp_target" ]]; then
                                printf 'misen bootstrap: %s transport did not create a directory\n' "$operation" >&2
                                rm -rf -- "$temp_root"
                                return 1
                            fi
                            if mv -T -- "$temp_target" "$target" 2>/dev/null || [[ -d "$target" ]]; then
                                :
                            elif [[ ! -e "$target" ]]; then
                                mv -- "$temp_target" "$target"
                            else
                                printf 'misen bootstrap: could not publish fetched snapshot\n' >&2
                                rm -rf -- "$temp_root"
                                return 1
                            fi
                        else
                            if [[ ! -f "$temp_target" ]]; then
                                printf 'misen bootstrap: %s transport did not create a file\n' "$operation" >&2
                                rm -rf -- "$temp_root"
                                return 1
                            fi
                            chmod 0600 -- "$temp_target" 2>/dev/null || true
                            mv -f -- "$temp_target" "$target"
                        fi
                        rm -rf -- "$temp_root"
                    }
                    """
                ),
                _shell_block(
                    f"""
                    snapshot_key={_quote(snapshot_key or "")}
                    project_dir="$store_root/snapshots/$snapshot_key"
                    fetch_transport snapshot "$snapshot_key" "$project_dir" directory

                    job_file_root="$store_root/job-files/{transport_key}"
                    payload_ref={_quote(payload)}
                    payload_path="$job_file_root/{payload_key}"
                    fetch_transport job-file "$payload_ref" "$payload_path" file

                    env_file_refs=({_array(env_files)})
                    env_file_keys=({_array(env_file_keys)})
                    env_file_paths=()
                    for i in "${{env_file_refs[@]+"${{!env_file_refs[@]}}"}}"; do
                        path="$job_file_root/${{env_file_keys[$i]}}"
                        fetch_transport job-file "${{env_file_refs[$i]}}" "$path" file
                        env_file_paths+=("$path")
                    done
                    """
                ),
            )
        )

    materialize = _shell_block(
        f"""
        materialize=(
            "$MISEN_UV_BIN" run --no-project
            --python {_quote(python_version)}
            --with "$misen_requirement"
            -m misen.utils.materialize_env
            --project-dir "$project_dir"
            --payload "$payload_path"
            --env-store-root "$store_root"
        )
        if (( ${{#env_file_paths[@]}} )); then
            materialize+=(--env-file "${{env_file_paths[@]}}")
        fi
        """
    )
    if transport_script is not None:
        materialize += '\nmaterialize+=(--snapshot-key "$snapshot_key")'
    if requires_pixi:
        materialize += '\nmaterialize+=(--pixi-bin "$MISEN_PIXI_BIN")'
    materialize += f'\nmaterialize+=({_array(worker_args)})\nexec "${{materialize[@]}}"'
    blocks.append(materialize)

    return "\n\n".join(blocks)
