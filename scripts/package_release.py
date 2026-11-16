"""Create a clean GitHub upload ZIP and SHA-256 manifest."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git", ".venv", ".pytest_cache", "__pycache__", "results", "private",
    "maldi_t3ss.egg-info",
}
SKIP_NAMES = {"config.yml", "RELEASE_MANIFEST.sha256"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def release_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or set(relative.parts).intersection(SKIP_PARTS):
            continue
        if path.name in SKIP_NAMES or path.suffix == ".pyc":
            continue
        yield path, relative


def main() -> None:
    files = list(release_files())
    manifest = "\n".join(f"{digest(path)}  {relative.as_posix()}" for path, relative in files) + "\n"
    (ROOT / "RELEASE_MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    archive = ROOT.parent / "maldi-t3ss-github-ready.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path, relative in files:
            bundle.write(path, Path("maldi-t3ss") / relative)
        bundle.writestr("maldi-t3ss/RELEASE_MANIFEST.sha256", manifest)
    print(f"Created {archive} with {len(files) + 1} files")


if __name__ == "__main__":
    main()

