#!/usr/bin/env bash
# 180 - READ-ONLY report: who holds the contiguous memory (CMA) that the TV picture needs.
# Why: since remote desktop came into use, the TV's login screen keeps failing to get screen buffers
# ("DRM_IOCTL_MODE_CREATE_DUMB failed: Cannot allocate memory": 1437 times by 18:40 on boot 46d8402b), and the
# kernel's counters say about 229 of the 256 MiB CMA pool is handed out and not given back. The remote-desktop
# session uses only the Mali's render node, which does not take CMA. Naming the holder needs root (debugfs, and
# other users' /proc entries).
# What it does: reads, and writes one report file:
#   1. the CMA pool: size, use, failed requests, largest free block
#   2. the TV display device (meson, ff900000.vpu): the programs that have it open, what is on screen (mode,
#      planes), and its framebuffers with the program that made each one and their sizes
#   3. every program's graphics-device and shared-buffer (dma-buf) handles, with sizes
#   4. recent kernel lines about CMA, and this boot's count of screen-buffer failures
# It changes nothing, loads nothing and restarts nothing. Nothing of the NPU (etnaviv) is opened or read: NPU
# handles are only counted from their /proc link names.
# Risk: none expected. Reading the display's debugfs files holds its locks for a moment.
# Run:    sudo bash root-steps/180-display-memory-report.sh
#         To test it without root: bash root-steps/180-display-memory-report.sh --check (prints only; the root-only
#         parts say "cannot read")
# Revert: nothing to revert. The report is ~/alta-npu/audit/cma-<date>-<time>.txt.
# Verify: the report ends with "end of report". Its summary says how much of the held memory the display's
#         framebuffers and the shared buffers account for.
set -uo pipefail  # a report: carry on past anything that cannot be read (no -e), and keep traces out of it (no -x)
here="$(cd "$(dirname "$0")/.." && pwd)"
check=0
[ "${1:-}" = --check ] && check=1
pg=$(getconf PAGESIZE)
mib() { echo $(( $1 * pg / 1048576 )); }  # pages -> MiB
sec() { printf '\n===== %s =====\n' "$*"; }
show() { if [ -r "$1" ]; then sed 's/^/  /' "$1"; else echo "  (cannot read $1)"; fi; }

report() {
  sec "1. CMA pool"
  echo "  $(date -Is), boot $(cut -c1-8 /proc/sys/kernel/random/boot_id), up $(cut -d' ' -f1 /proc/uptime) s"
  grep -E '^Cma(Total|Free):' /proc/meminfo | sed 's/^/  /'
  local held=0 d a r f
  for d in /sys/kernel/mm/cma/*/; do
    [ -r "$d/alloc_pages_success" ] || continue
    a=$(cat "$d/alloc_pages_success"); r=$(cat "$d/release_pages_success"); f=$(cat "$d/alloc_pages_fail")
    held=$((held + a - r))
    echo "  $(basename "$d"): handed out $(mib "$a") MiB since boot, given back $(mib "$r") MiB, held now $(mib $((a - r))) MiB; failed requests: $f pages (retries count again)"
  done
  for d in /sys/kernel/debug/cma/*/; do
    if [ ! -r "$d/used" ]; then echo "  (cannot read $d: debugfs needs root)"; continue; fi
    echo "  debugfs $(basename "$d"): pool $(mib "$(cat "$d/count")") MiB, used $(mib "$(cat "$d/used")") MiB, largest free block $(mib "$(cat "$d/maxchunk")") MiB"
  done

  sec "2. TV display device"
  local card dbg res fb_total=0
  card=$(readlink -f /dev/dri/by-path/platform-ff900000.vpu-card 2>/dev/null || true)
  if [ -z "$card" ] || [ ! -c "$card" ]; then
    echo "  display device (platform-ff900000.vpu-card) not found"
  else
    dbg="/sys/kernel/debug/dri/$((16#$(stat -c %T "$card")))"
    [ -e "$dbg" ] || [ ! -e /sys/kernel/debug/dri/ff900000.vpu ] || dbg=/sys/kernel/debug/dri/ff900000.vpu
    echo "  device $card, debugfs $dbg"
    if [ -r "$dbg/name" ] && ! grep -q meson "$dbg/name"; then
      echo "  $dbg/name does not say meson, so nothing more is read there: $(cat "$dbg/name")"
    else
      echo "--- name"; show "$dbg/name"
      echo "--- programs that have it open"; show "$dbg/clients"
      echo "--- on screen now"; show "$dbg/state"
      echo "--- framebuffers ('allocated by' is the program that made each one)"; show "$dbg/framebuffer"
      if [ -r "$dbg/framebuffer" ]; then
        # Each buffer is counted once, keyed by its DMA address (or its mmap offset) when a framebuffer reuses one.
        res=$(awk 'function flush() { if (sz != "") { k = (da != "" ? da : (st != "" && st !~ /^0+$/ ? "s" st : "u" NR)); if (!(k in seen)) { seen[k] = 1; by[who] += sz; tot += sz } } sz = ""; da = ""; st = "" }
                   /^framebuffer\[/ { flush() }
                   /allocated by =/ { s = $0; sub(/.*allocated by = /, "", s); who = s }
                   /obj\[[0-9]+\]:/ { flush() }
                   /^[ \t]+start=/ { s = $0; sub(/.*start=/, "", s); st = s }
                   /^[ \t]+size=[0-9]+[ \t]*$/ { s = $0; sub(/.*size=/, "", s); sz = s + 0 }
                   /^[ \t]+dma_addr=/ { s = $0; sub(/.*dma_addr=/, "", s); da = s }
                   END { flush(); for (w in by) printf "  %-20s %7.1f MiB\n", w, by[w] / 1048576; printf "FBTOTAL %d\n", tot }' "$dbg/framebuffer")
        echo "--- framebuffer memory by the program that made it"
        echo "$res" | grep -v '^FBTOTAL'
        fb_total=$(echo "$res" | awk '/^FBTOTAL/ {print $2}')
      fi
    fi
  fi

  sec "3. Graphics-device and shared-buffer handles of every program"
  local mali_c mali_r npu_c npu_r rows shared_total
  mali_c=$(readlink -f /dev/dri/by-path/platform-ffe40000.gpu-card 2>/dev/null || true)
  mali_r=$(readlink -f /dev/dri/by-path/platform-ffe40000.gpu-render 2>/dev/null || true)
  npu_c=$(readlink -f /dev/dri/by-path/platform-etnaviv-card 2>/dev/null || true)
  npu_r=$(readlink -f /dev/dri/by-path/platform-etnaviv-render 2>/dev/null || true)
  # Only the link names of the open files are read here. fdinfo is read for shared buffers only, and the shared-buffer
  # core prints it without asking the driver that made the buffer.
  rows=$(find /proc/[0-9]*/fd -mindepth 1 -maxdepth 1 -printf '%p|%l\n' 2>/dev/null | while IFS='|' read -r fd t; do
    pid=${fd#/proc/}; pid=${pid%%/*}; n=${fd##*/}
    case "$t" in
      /dev/dri/*)
        case "$t" in "$card") dev=TV-display ;; "$mali_c"|"$mali_r") dev=Mali ;; "$npu_c"|"$npu_r") dev=NPU ;; *) dev=other ;; esac
        echo "dri|$pid|$dev|$t" ;;
      /dmabuf:*|anon_inode:dmabuf*)
        info="/proc/$pid/fdinfo/$n"
        echo "buf|$pid|$(awk '/^ino:/ {print $2}' "$info" 2>/dev/null)|$(awk '/^size:/ {print $2}' "$info" 2>/dev/null)" ;;
      /dev/video*|/dev/dma_heap/*) echo "dev|$pid|$t" ;;
    esac
  done)
  echo "--- programs with a graphics device open (number of handles, device)"
  echo "$rows" | awk -F'|' '$1 == "dri" { c[$2 "|" $3 "|" $4]++ } END { for (k in c) print k "|" c[k] }' | sort -t'|' -k1,1n |
    while IFS='|' read -r pid dev node cnt; do
      printf '  %-7s %-18s %-11s %-20s %3s handles\n' "$pid" "$(cat "/proc/$pid/comm" 2>/dev/null)" "$dev" "$node" "$cnt"
    done
  echo "--- shared buffers (dma-buf) by program; a buffer shared by several programs is listed under each"
  echo "$rows" | awk -F'|' '$1 == "buf" && $3 != "" { k = $2 "|" $3; if (!(k in s)) { s[k] = 1; n[$2]++; sz[$2] += $4 } }
      END { for (p in n) printf "%s|%d|%.1f\n", p, n[p], sz[p] / 1048576 }' | sort -t'|' -k1,1n |
    while IFS='|' read -r pid cnt m; do
      printf '  %-7s %-18s %4s buffers %8s MiB\n' "$pid" "$(cat "/proc/$pid/comm" 2>/dev/null)" "$cnt" "$m"
    done
  shared_total=$(echo "$rows" | awk -F'|' '$1 == "buf" && $3 != "" && !($3 in a) { a[$3] = 1; n++; t += $4 } END { printf "%d buffers, %.1f MiB", n, t / 1048576 }')
  echo "  all programs together: $shared_total"
  echo "--- video devices and DMA heaps held open"
  if echo "$rows" | grep -q '^dev|'; then
    echo "$rows" | awk -F'|' '$1 == "dev" {print $2 "|" $3}' | sort -u | while IFS='|' read -r pid node; do
      printf '  %-7s %-18s %s\n' "$pid" "$(cat "/proc/$pid/comm" 2>/dev/null)" "$node"
    done
  else
    echo "  none"
  fi

  sec "4. Kernel lines about CMA, and screen-buffer failures"
  journalctl -k -b --no-pager -q 2>/dev/null | grep -iE 'cma|dma_alloc|alloc failed' | tail -15 | sed 's/^/  /'
  echo "  screen-buffer failures (CREATE_DUMB) this boot: $(journalctl -b --no-pager -q 2>/dev/null | grep -c 'CREATE_DUMB failed')"
  loginctl list-sessions --no-legend 2>/dev/null | sed 's/^/  session: /'

  sec "5. Summary"
  echo "  held in the CMA pool now: $(mib "$held") MiB (kernel counters)"
  echo "  the TV display's framebuffers: $(( ${fb_total:-0} / 1048576 )) MiB (0 if debugfs could not be read)"
  echo "  shared buffers (dma-buf; these can be CMA or ordinary memory): $shared_total"
  echo "  If the framebuffers come close to what is held, the program named in section 2 holds the pool. If not, the"
  echo "  rest is held by kernel drivers that this report cannot name."
  echo "end of report"
}

if [ "$check" -eq 1 ]; then
  report
  exit 0
fi
[ "$(id -u)" -eq 0 ] || { echo "180: run it with sudo (or with --check to test it without root)" >&2; exit 1; }
out="$here/audit/cma-$(date +%Y%m%d-%H%M%S).txt"
mkdir -p "$here/audit"
"$here/scripts/progress" 180 STARTING "read-only report of who holds the CMA display memory; output $out"
report 2>&1 | tee "$out"
[ -n "${SUDO_USER:-}" ] && chown "$SUDO_USER": "$out"
"$here/scripts/progress" 180 DONE "read-only CMA report: $out"
echo "Done: report in $out"
