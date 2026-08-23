"""Locate or lazily install the standalone ``uv`` executable."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from functools import cache
from pathlib import Path
from textwrap import dedent

UV_BIN_ENV = "MISEN_UV_BIN"
UV_AUTO_INSTALL_ENV = "MISEN_UV_AUTO_INSTALL"
UV_VERSION = "0.12.3"
UV_MIN_VERSION = "0.11.15"
_UV_INSTALLER_URL = f"https://astral.sh/uv/{UV_VERSION}/install.sh"
_UV_MIN_MAJOR, _UV_MIN_MINOR, _UV_MIN_PATCH = UV_MIN_VERSION.split(".")


# This resolver is embedded in worker bootstraps and also used on the
# submitter. The versioned Astral installer selects the platform artifact and
# verifies its checksum where ``sha256sum`` is available. Installing into a
# temporary directory keeps the shared cache free of partial executables.
_UV_RESOLVER = dedent(
    rf"""
    uv_is_supported() {{
        local output
        output="$("$1" --version 2>/dev/null)" || return 1
        [[ "$output" =~ ^uv[[:space:]]+([0-9]+)\.([0-9]+)\.([0-9]+) ]] || return 1
        (( 10#${{BASH_REMATCH[1]}} > {_UV_MIN_MAJOR} ||
            (10#${{BASH_REMATCH[1]}} == {_UV_MIN_MAJOR} &&
                (10#${{BASH_REMATCH[2]}} > {_UV_MIN_MINOR} ||
                    (10#${{BASH_REMATCH[2]}} == {_UV_MIN_MINOR} &&
                        10#${{BASH_REMATCH[3]}} >= {_UV_MIN_PATCH}))) ))
    }}

    uv_is_pinned() {{
        local output
        output="$("$1" --version 2>/dev/null)" || return 1
        [[ "$output" == "uv {UV_VERSION}" || "$output" == "uv {UV_VERSION} "* ]]
    }}

    resolve_uv() {{
        local preferred="$1"
        local install_dir="$2"
        local resolved temp_dir

        for resolved in "${{{UV_BIN_ENV}:-}}" "$preferred"; do
            if [[ -n "$resolved" ]] && uv_is_supported "$resolved"; then
                printf '%s\n' "$resolved"
                return
            fi
        done
        if resolved="$(command -v uv 2>/dev/null)" && uv_is_supported "$resolved"; then
            printf '%s\n' "$resolved"
            return
        fi
        if uv_is_pinned "$install_dir/uv"; then
            printf '%s\n' "$install_dir/uv"
            return
        fi
        if [[ "${{{UV_AUTO_INSTALL_ENV}:-1}}" == 0 ]]; then
            printf 'misen bootstrap: uv >= {UV_MIN_VERSION} is required; install it or set {UV_BIN_ENV}\n' >&2
            return 127
        fi

        if command -v curl >/dev/null 2>&1; then
            download_uv() {{
                curl --proto '=https' --connect-timeout 15 --max-time 60 --retry 2 -LsSf {_UV_INSTALLER_URL}
            }}
        elif command -v wget >/dev/null 2>&1; then
            download_uv() {{
                wget --timeout=60 --tries=2 -qO- {_UV_INSTALLER_URL}
            }}
        else
            printf 'misen bootstrap: downloading uv requires curl or wget; install uv or set {UV_BIN_ENV}\n' >&2
            return 127
        fi

        mkdir -p -- "$install_dir" || return 1
        temp_dir="$(mktemp -d "$install_dir/.install.XXXXXX")" || return 1
        if ! (
            trap 'rm -rf -- "$temp_dir"' EXIT
            download_uv |
                AUTH_TOKEN= UV_GITHUB_TOKEN= UV_INSTALL_DIR= CARGO_DIST_FORCE_INSTALL_DIR= \
                UV_DOWNLOAD_URL= INSTALLER_DOWNLOAD_URL= \
                UV_INSTALLER_GHE_BASE_URL= UV_INSTALLER_GITHUB_BASE_URL= \
                UV_UNMANAGED_INSTALL="$temp_dir" sh >&2 &&
                uv_is_pinned "$temp_dir/uv" &&
                mv -f -- "$temp_dir/uv" "$install_dir/uv" &&
                uv_is_pinned "$install_dir/uv"
        ); then
            if uv_is_pinned "$install_dir/uv"; then
                printf '%s\n' "$install_dir/uv"
                return
            fi
            printf 'misen bootstrap: uv installer failed; install uv or set {UV_BIN_ENV}\n' >&2
            return 1
        fi
        printf '%s\n' "$install_dir/uv"
    }}
    """
).strip()


def ensure_uv_script(preferred: str, store_root: Path) -> str:
    """Render shell that resolves uv and exports its path as ``MISEN_UV_BIN``."""
    resolution = dedent(
        f"""\
        uv_store_root={shlex.quote(str(store_root))}
        uv_platform="$(uname -s)-$(uname -m)"
        uv_install_dir="$uv_store_root/tools/uv/{UV_VERSION}/$uv_platform"
        MISEN_UV_BIN="$(resolve_uv {shlex.quote(preferred)} "$uv_install_dir")"
        export MISEN_UV_BIN
        """
    ).strip()
    return f"{_UV_RESOLVER}\n\n{resolution}"


def _managed_store_root() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "misen"


@cache
def find_or_install_uv() -> str:
    """Return a supported uv executable, downloading the pinned fallback if needed."""
    bash = shutil.which("bash")
    if bash is None:
        msg = "Misen needs Bash plus curl or wget to install uv automatically; install uv or set MISEN_UV_BIN."
        raise RuntimeError(msg)
    script = f"set -euo pipefail\n{ensure_uv_script('', _managed_store_root())}\nprintf '%s\\n' \"$MISEN_UV_BIN\""
    try:
        result = subprocess.run(  # noqa: S603
            [bash, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout).strip()
        msg = detail or f"Could not locate or install uv >= {UV_MIN_VERSION}."
        raise RuntimeError(msg) from e
    path = result.stdout.strip()
    if not path:
        msg = "uv resolver returned no executable path."
        raise RuntimeError(msg)
    return path
