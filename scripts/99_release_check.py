"""Fail before release if common identifiers, local paths, caches or oversized files are present."""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__", "results", "private"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".cff", ".csv", ".json"}
PATTERNS = {
    "absolute Windows drive path": re.compile(r"(?i)\b[A-Z]:[\\/](?!/)") ,
    "patient name field": re.compile(r"姓名|患者姓名"),
    "admission/medical record field": re.compile(r"住院号|病历号|身份证号|手机号"),
}
IDENTIFIER_SUFFIXES = {".csv", ".json", ".yml", ".yaml"}


def main() -> None:
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or set(path.relative_to(ROOT).parts).intersection(SKIP_PARTS):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.name == "config.yml":  # local, gitignored configuration used for private reproduction
            continue
        if path.stat().st_size > 90 * 1024 * 1024:
            failures.append(f"oversized file: {path.relative_to(ROOT)}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if label != "absolute Windows drive path" and path.suffix.lower() not in IDENTIFIER_SUFFIXES:
                continue
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(ROOT)}")
    if failures:
        print("Release check failed:")
        print("\n".join(f"- {item}" for item in failures))
        sys.exit(1)
    print("Release check passed: no common direct identifiers, absolute drive paths, or oversized files found.")


if __name__ == "__main__":
    main()

