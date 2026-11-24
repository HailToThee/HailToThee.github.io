#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import os
import shutil
from pathlib import Path

SRC = Path("/home/xjc/Projects/mynotes")
DST = Path("/home/xjc/Projects/HailToThee.github.io/content")
EXCLUDE_DIRS = {".git", ".obsidian", ".vscode", "__pycache__", "node_modules"}
ALLOWED_EXTS = {".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf"}

if DST.exists():
    for item in DST.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
else:
    DST.mkdir(parents=True, exist_ok=True)

copied = 0
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    rel_root = Path(root).relative_to(SRC)
    for fname in files:
        src_file = Path(root) / fname
        if src_file.suffix.lower() not in ALLOWED_EXTS:
            continue
        dst_file = DST / rel_root / fname
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1

readme = DST / "README.md"
index = DST / "index.md"
if readme.exists() and not index.exists():
    shutil.copy2(readme, index)

print(f"Synced {copied} files to {DST}")
PY
