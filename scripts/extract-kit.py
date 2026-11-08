#!/usr/bin/env python3
"""extract-kit.py - write every code block in the handoff that is preceded by an HTML comment
line of the form  file: RELATIVE/PATH [mode=NNN]  to disk under DEST. Existing files are kept unless --force.
Usage: python3 scripts/extract-kit.py ALTA_NPU_HANDOFF.md [DEST] [--force]"""
import os
import pathlib
import re
import sys

args = [a for a in sys.argv[1:] if a != "--force"]
force = "--force" in sys.argv
src = pathlib.Path(args[0] if args else "ALTA_NPU_HANDOFF.md")
dst = pathlib.Path(args[1] if len(args) > 1 else ".")
FENCE = "`" * 3
pat = re.compile(r"^<!-- file: (\S+)(?: mode=([0-7]{3,4}))? -->\n" + FENCE + r"[^\n]*\n(.*?)\n" + FENCE + r"[ \t]*$",
                 re.S | re.M)
n = 0
for rel, mode, body in pat.findall(src.read_text()):
    if rel.startswith("/") or ".." in pathlib.PurePosixPath(rel).parts:
        sys.exit(f"refusing unsafe path: {rel}")
    p = dst / rel
    if p.exists() and not force:
        print(f"skip (exists): {p}")
        continue
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body + "\n")
    if mode:
        os.chmod(p, int(mode, 8))
    n += 1
    print(f"wrote {p}")
print(f"{n} file(s) written to {dst.resolve()}")
