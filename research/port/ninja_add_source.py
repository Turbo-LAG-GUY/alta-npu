#!/usr/bin/env python3
"""Add a new etnaviv source file to mesa-compute/build/build.ninja without letting meson regenerate.

meson must never regenerate mesa-compute/build: its coredata points at subprojects in Jordan's
~/Documents/AI_Projects/mesa. So a new .c file goes into src/gallium/drivers/etnaviv/meson.build (for the patch) and,
with this script, into build.ninja by hand: the compile edge of etnaviv_context.c is copied with the new name and the
object is added to the libetnaviv.a link line. The script then touches build.ninja so it is newer than meson.build.
Check afterwards: `ninja -n -C mesa-compute/build src/gallium/drivers/etnaviv/libetnaviv.a | grep -c Regenerat` = 0.

usage: ninja_add_source.py etnaviv_foo.c [--build /home/jordan/Documents/AI_Projects/mesa-compute/build]
"""
import argparse
import os
import shutil
import sys

ap = argparse.ArgumentParser()
ap.add_argument("source", help="file name in src/gallium/drivers/etnaviv, e.g. etnaviv_compute.c")
ap.add_argument("--build", default="/home/jordan/Documents/AI_Projects/mesa-compute/build")
a = ap.parse_args()

if "mesa-axi-sram" in os.path.realpath(a.build) or "/mesa/" in os.path.realpath(a.build) + "/":
    sys.exit("refusing: only for the port tree, never the kit tree or Jordan's mesa repo")

ninja = os.path.join(a.build, "build.ninja")
lines = open(ninja).read().split("\n")
obj_dir = "src/gallium/drivers/etnaviv/libetnaviv.a.p/"
new_obj = obj_dir + a.source + ".o"
if any(l.startswith("build " + new_obj + ":") for l in lines):
    sys.exit("already in build.ninja: " + new_obj)

start = [i for i, l in enumerate(lines) if l.startswith("build " + obj_dir + "etnaviv_context.c.o: c_COMPILER ")]
if len(start) != 1:
    sys.exit("template edge not found exactly once")
i = j = start[0]
block = []
while lines[j] != "":
    block.append(lines[j])
    j += 1
stem = a.source[:-2] if a.source.endswith(".c") else a.source
edge = [l.replace("etnaviv_context.c", a.source).replace("etnaviv_context", stem) for l in block]
lines[j:j] = [""] + edge

lib = [k for k, l in enumerate(lines) if l.startswith("build src/gallium/drivers/etnaviv/libetnaviv.a: STATIC_LINKER ")]
tok = " " + obj_dir + "etnaviv_context.c.o"
if len(lib) != 1 or lines[lib[0]].count(tok) != 1:
    sys.exit("libetnaviv.a link line not found")
lines[lib[0]] = lines[lib[0]].replace(tok, tok + " " + new_obj)

backup = ninja + ".before-" + stem
shutil.copy2(ninja, backup)
open(ninja, "w").write("\n".join(lines))
os.utime(ninja)
print("added %s (backup %s); now run ninja -n and check there is no 'Regenerating'" % (new_obj, backup))
