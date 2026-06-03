"""Remove generated report and cache folders without touching user inputs."""

from __future__ import annotations

from pathlib import Path
import shutil

GENERATED_DIRECTORIES = (
    Path("output"),
    Path("cache"),
    Path("data/processed"),
    Path(".pytest_cache"),
    Path("src/wc_predictor.egg-info"),
    Path("src/wc_predictor/__pycache__"),
    Path("scripts/__pycache__"),
    Path("tests/__pycache__"),
)
LEGACY_PASTE_DIRECTORIES = (
    Path("data/raw/oddsportal_combined_pastes"),
    Path("data/raw/oddsportal_pastes"),
)


def _resolved_cleanup_targets(workspace: Path) -> tuple[Path, ...]:
    """Resolve the allowlisted generated directories and reject path escapes."""

    workspace = workspace.resolve()
    targets: list[Path] = []
    for relative_path in GENERATED_DIRECTORIES:
        target = (workspace / relative_path).resolve()
        if target.parent == workspace or workspace in target.parents:
            targets.append(target)
            continue
        raise RuntimeError(f"Refusing to remove path outside workspace: {target}")
    return tuple(targets)


def clean_generated_outputs(workspace: str | Path = ".") -> tuple[Path, ...]:
    """Delete generated directories and return the paths that existed."""

    workspace = Path(workspace).resolve()
    removed: list[Path] = []
    for target in _resolved_cleanup_targets(workspace):
        if target.exists():
            shutil.rmtree(target)
            removed.append(target)
    return tuple(removed)


def find_legacy_paste_files(workspace: str | Path = ".") -> tuple[Path, ...]:
    """Return retired OddsPortal paste files so cleanup can warn before removal."""

    workspace = Path(workspace).resolve()
    files: list[Path] = []
    for relative_path in LEGACY_PASTE_DIRECTORIES:
        folder = (workspace / relative_path).resolve()
        if not (folder.parent == workspace or workspace in folder.parents):
            raise RuntimeError(f"Refusing to inspect path outside workspace: {folder}")
        if folder.exists():
            files.extend(path for path in folder.rglob("*") if path.is_file())
    return tuple(sorted(files))


def main() -> None:
    """Clean generated folders below the current project directory."""

    legacy_paste_files = find_legacy_paste_files()
    if legacy_paste_files:
        print("WARNING: Legacy OddsPortal paste files found. They will not be deleted automatically:")
        for path in legacy_paste_files:
            print(f"  - {path}")
        print("Move any files you still need into input/odds/ before removing legacy folders.")
        print()
    removed = clean_generated_outputs()
    print("Generated Output Cleanup")
    print(f"- Directories removed: {len(removed)}")
    for path in removed:
        print(f"  - {path}")
    if not removed:
        print("  - none")


if __name__ == "__main__":
    main()
