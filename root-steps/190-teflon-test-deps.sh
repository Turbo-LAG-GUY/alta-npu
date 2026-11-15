#!/usr/bin/env bash
# 190 - install the packages needed to BUILD Mesa's Teflon test suite (test_teflon).
# Why: the upstream etnaviv/ml series (upstream/etnaviv-ml-fixes) has passed our own ladders on the NPU, but Mesa's
# own test suite has not been run. It cannot be built here: gtest, cmake and the system Python's flatbuffers
# module are missing (checked 2026-11-15).
# What it does: one apt-get install of three packages from Ubuntu 24.04 (7 packages with their dependencies,
# about 40 MB installed):
#   libgtest-dev          the test framework test_teflon is written with
#   python3-flatbuffers   for meson's Python: Mesa's script that splits the model zoo into test models needs it
#   cmake                 only to build TensorFlow Lite afterwards (scripts/build-tflite.sh, no root)
# It does not touch the kernel, etnaviv, udev, Mesa or any library already on the board, and runs nothing on
# the NPU. No reboot.
# Risk: low. Three development packages; nothing already installed is upgraded or removed (apt-get -s, 2026-11-15:
# "0 upgraded, 7 newly installed, 0 to remove").
# Run:    sudo bash ~/alta-npu/root-steps/190-teflon-test-deps.sh
#         To see what apt would do without changing anything: bash root-steps/190-teflon-test-deps.sh --check
# Revert: sudo apt-get remove libgtest-dev googletest python3-flatbuffers cmake cmake-data libjsoncpp25 librhash0
# Verify: the script ends with "190 done"; then: cmake --version; pkg-config --modversion gtest;
#         python3 -c "import flatbuffers"
# Next (no root): bash ~/alta-npu/scripts/build-tflite.sh
set -euo pipefail
pkgs="libgtest-dev python3-flatbuffers cmake"

if [ "${1:-}" = --check ]; then
   apt-get -s install --no-install-recommends $pkgs | grep -E '^(Inst|Remv|Conf)|upgraded' || true
   exit 0
fi
[ "$(id -u)" = 0 ] || { echo "run with sudo: sudo bash $0"; exit 1; }

# Refuse if apt would remove or upgrade anything: this step must only add packages.
sim=$(apt-get -s install --no-install-recommends $pkgs)
if echo "$sim" | grep -q '^Remv'; then echo "apt wants to REMOVE packages; stopping:"; echo "$sim" | grep '^Remv'; exit 1; fi
if ! echo "$sim" | grep -q '^0 upgraded'; then echo "apt wants to upgrade packages; stopping:"; echo "$sim" | grep upgraded; exit 1; fi

apt-get install -y --no-install-recommends $pkgs

cmake --version | head -1
pkg-config --modversion gtest
python3 -c "import flatbuffers; print('python3 flatbuffers', flatbuffers.__version__)"
echo "190 done"
