from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_copied_targets = []


def copy_collections(
    collections: dict, source_base: Path, target_base: Path, verbose: bool = False
) -> None:
    """
    Copy folders based on a collections config dictionary, similar to ``sphinxcontrib-collections``.

    This function iterates through the provided ``collections`` dictionary, copying folders from the specified
    ``source_base`` to the ``target_base`` according to each collection's configuration. If the target directory
    already exists, it will be removed before copying.

    Parameters
    ----------
    collections : dict
        Dictionary where each key maps to a collection config with:
            - ``source`` (str | Path): Relative path from ``source_base`` to copy from.
            - ``target`` (str | Path): Relative path from ``target_base`` to copy to.
            - ``ignore`` (list[str], optional): Patterns to ignore during copy.
    source_base : Path
        Base directory for the source paths.
    target_base : Path
        Base directory for the target paths.

    Returns
    -------
    None
        This function does not return a value.

    .. Note::
        - If the target directory exists, it will be deleted before copying.
        - The ``ignore`` patterns use the same syntax as ``shutil.ignore_patterns``.

    .. Warning::
        Use with caution: all contents of the target directory will be removed prior to copying.

    Examples
    --------
    >>> collections = {
    ...     "notebooks": {
    ...         "source": "src_folder",
    ...         "target": "dst_folder",
    ...         "ignore": ["*.pyc", "__pycache__"],
    ...     }
    ... }
    >>> copy_collections(
    ...     collections, Path("/tmp/source"), Path("/tmp/target")
    ... )
    """
    global _copied_targets
    for name, cfg in collections.items():
        src = source_base / cfg["source"]
        dst = target_base / cfg["target"]
        ignore_patterns = shutil.ignore_patterns(*cfg.get("ignore", []))

        if dst.exists():
            shutil.rmtree(dst)
            if verbose:
                print(f"[copy_collections] Removed existing: {dst}")

        shutil.copytree(src, dst, ignore=ignore_patterns)
        _copied_targets.append(dst)
        if verbose:
            print(f"[copy_collections] Copied {src} → {dst}")


def clean_copied(verbose: bool = False):
    global _copied_targets
    for path in _copied_targets:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            if verbose:
                print(f"[copy_collections] Cleaned up: {path}")
    _copied_targets.clear()


def get_poetry_version(default="0.0.0"):
    try:
        output = subprocess.check_output(["poetry", "version"]).decode().strip()
        return output.split()[1]  # output is like "myproject 1.2.3"
    except Exception:
        return default
