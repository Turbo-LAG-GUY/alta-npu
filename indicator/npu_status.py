#!/usr/bin/python3
"""npu-status: a read-only status indicator for the Amlogic A311D NPU.

Shows whether the NPU is working, how busy it is and which programs are using
it: as an icon in the GNOME top bar, as a live terminal view, or as a one-off
report.

It only reads kernel status files. It never opens the NPU device or sends it
work, so watching the NPU cannot load or disturb it.

    npu_status.py              top-bar indicator
    npu_status.py --watch      live view in the terminal
    npu_status.py --once       one report (add --json for JSON)
    npu_status.py --install    start the indicator at login (--uninstall)

Where the numbers come from:

    state, active %  power/runtime_status and power/runtime_active_time of the
                     NPU's sysfs device. The driver powers the NPU up for jobs
                     and back down 200 ms after the last one, so "active" is
                     powered-up time. Under sustained inference it matches real
                     use; a lone short job reads high.
                     If the NPU is kept powered on (power/control = on, as
                     alta-npu root step 100 does), powered-up time says nothing,
                     so the state comes from interrupts and the top bar shows
                     interrupts/s instead of a percentage.
    interrupts/s     the NPU's line in /proc/interrupts. The NPU raises an
                     interrupt as it finishes work.
    clock            the NPU's thermal cooling device. Each throttle step halves
                     the clock.
    temperature      /sys/class/thermal/thermal_zone*.
    programs         /proc/<pid>/fd entries pointing at the NPU's DRM nodes.
                     Without root, only your own processes are visible.
    AXI SRAM         /sys/module/etnaviv/parameters/axi_sram: the alta driver's
                     switch for the NPU's 1 MiB on-chip memory. Whether a program
                     asked for it is its ETNA_ML_AXI_SRAM variable.
    shader core      the NPU's programmable core. Programs opt in per operation
                     with ETNA_ML_SHADER (compute port, mesa-compute); read from
                     /proc/<pid>/environ, which holds the values the program was
                     started with.
    Teflon           which Mesa tree's libteflon.so a program has mapped
                     (/proc/<pid>/maps).
    hang             a kernel hang dump waiting in /sys/class/devcoredump (it is
                     deleted after about 5 minutes), and the kit's runs/STOP file,
                     which blocks NPU jobs until a reboot.
    job queue        /sys/module/etnaviv/parameters/hw_job_*: how many jobs wait in
                     the NPU at once, and whether the driver dropped that to 1 after
                     a recovered hang (alta kernel patches 0006 and 0007). The menu's
                     "Job queue…" opens npu_control.py, the window that changes it.

The indicator writes nothing to the NPU. It does write the job-queue settings you
saved in that window, once per boot when it starts ("apply when I log in"), and
when the driver drops the queue to 1 after a hang it tells you and opens the
window so you can put it back, keep it at 1 or reboot (npu_queue.py).
"""

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

try:
    import npu_queue  # job-queue settings (alta patches 0006/0007); ships beside this file
except ImportError:
    npu_queue = None

APP_ID = "npu-status"
ICON_VERSION = "v1"  # bump when the icon design changes: the shell caches icons by name
LEVELS = 4           # fill steps in the top-bar icon
HISTORY = 60         # samples kept for the terminal sparkline
ONCHIP_CACHE_BYTES = 512 * 1024  # this NPU's on-chip SRAM (GC8000 r7120, kernel etnaviv_hwdb.c)
AXI_SRAM_BYTES = 1024 * 1024     # its AXI SRAM (same hwdb entry); used only by programs that ask for it
AXI_SRAM_SWITCH = "/sys/module/etnaviv/parameters/axi_sram"  # alta kernel patch 0005; absent upstream
DEVCOREDUMP_DIR = "/sys/class/devcoredump"
# The alta-npu kit's npu-run writes this after an NPU hang and then refuses NPU jobs until a reboot.
STOP_FILE = os.path.join(os.path.expanduser("~"), "alta-npu", "runs", "STOP")

# Teflon's on-disk weight cache stops growing with less free space than this. Keep it
# equal to NPU_CACHE_NO_GROW_BELOW in Mesa's etnaviv_ml_disk_cache.h (alta patch 0004).
DISK_LOW_BYTES = 10 * 1024 ** 3
DISK_REARM_BYTES = 11 * 1024 ** 3  # warn again only after free space has been back above this
DISK_CHECK_SECONDS = 60

ROOT = "/"          # tests point this at a fake /sys and /proc (--root)


def host_path(path):
    return path if ROOT == "/" else os.path.join(ROOT, path.lstrip("/"))


def read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def read_int(path):
    try:
        return int(read_text(path))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Finding the NPU


@dataclass
class Npu:
    name: str             # kernel device name, e.g. "ff100000.npu"
    sysfs: str            # its sysfs directory
    nodes: frozenset      # its DRM device nodes, e.g. {"/dev/dri/renderD128"}
    cooling: str | None   # its thermal cooling device directory
    board: str            # board model from the device tree


def find_npu():
    """Return the NPU core bound to the etnaviv driver, or None."""
    drv = host_path("/sys/bus/platform/drivers/etnaviv-gpu")
    try:
        entries = sorted(os.listdir(drv))
    except OSError:
        return None
    cores = [e for e in entries
             if e != "module" and os.path.isdir(os.path.join(drv, e, "power"))]
    if not cores:
        return None
    # On SoCs where etnaviv drives both a GPU and an NPU, prefer the NPU.
    name = next((c for c in cores if "npu" in c), cores[0])
    board = read_text(host_path("/proc/device-tree/model")) or ""
    return Npu(name=name,
               sysfs=os.path.realpath(os.path.join(drv, name)),
               nodes=_drm_nodes(),
               cooling=_cooling_device(name),
               board=board.strip("\0 "))


def _drm_nodes():
    master = host_path("/sys/bus/platform/drivers/etnaviv")
    nodes = set()
    try:
        devices = os.listdir(master)
    except OSError:
        return frozenset()
    for dev in devices:
        try:
            names = os.listdir(os.path.join(master, dev, "drm"))
        except OSError:
            continue
        nodes.update("/dev/dri/" + n for n in names if n.startswith(("card", "renderD")))
    return frozenset(nodes)


def _cooling_device(name):
    base = host_path("/sys/class/thermal")
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return None
    for e in entries:
        if e.startswith("cooling_device") and read_text(os.path.join(base, e, "type")) == name:
            return os.path.join(base, e)
    return None


def _thermal_zones():
    """[(label, temp file)] for every thermal zone, e.g. ("cpu", ".../temp")."""
    base = host_path("/sys/class/thermal")
    try:
        entries = sorted((e for e in os.listdir(base) if e.startswith("thermal_zone")),
                         key=lambda e: (len(e), e))
    except OSError:
        return []
    zones = []
    for e in entries:
        label = read_text(os.path.join(base, e, "type")) or e
        for suffix in ("-thermal", "_thermal"):
            label = label.removesuffix(suffix)
        zones.append((label, os.path.join(base, e, "temp")))
    return zones


# --------------------------------------------------------------------------
# Programs that have the NPU open


@dataclass
class Client:
    pid: int
    name: str
    cmd: str
    mapped: int  # bytes of NPU buffers mapped into the process
    shader: str | None = None  # ETNA_ML_SHADER it was started with, e.g. "transpose,softmax"
    axi: str | None = None     # ETNA_ML_AXI_SRAM it was started with, e.g. "all"
    teflon: str | None = None  # Mesa tree of the libteflon.so it has mapped, e.g. "mesa-compute"


def find_clients(nodes):
    """Processes holding one of the NPU's device nodes open (that we may inspect)."""
    proc = host_path("/proc")
    me = os.getpid()
    found = []
    try:
        pids = [p for p in os.listdir(proc) if p.isdigit() and int(p) != me]
    except OSError:
        return found
    for pid in pids:
        base = os.path.join(proc, pid)
        try:
            with os.scandir(os.path.join(base, "fd")) as fds:
                if not any(_readlink(fd.path) in nodes for fd in fds):
                    continue
        except OSError:  # gone, or another user's process
            continue
        mapped, teflon = _read_maps(os.path.join(base, "maps"), nodes)
        env = _read_environ(os.path.join(base, "environ"), ("ETNA_ML_SHADER", "ETNA_ML_AXI_SRAM"))
        found.append(Client(pid=int(pid),
                            name=read_text(os.path.join(base, "comm")) or "?",
                            cmd=_describe_cmdline(os.path.join(base, "cmdline")),
                            mapped=mapped,
                            shader=env.get("ETNA_ML_SHADER") or None,
                            axi=env.get("ETNA_ML_AXI_SRAM") or None,
                            teflon=teflon))
    return sorted(found, key=lambda c: c.pid)


def _readlink(path):
    try:
        return os.readlink(path)
    except OSError:
        return None


def _describe_cmdline(path):
    """'python3 -u /x/y/run_llm.py -m m.tflite' -> 'python3 run_llm.py'."""
    try:
        with open(path, "rb") as f:
            args = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return ""
    if not args:
        return ""
    words = [os.path.basename(args[0])]
    if re.fullmatch(r"(python|perl|ruby|node|bash|sh)[\d.]*", words[0]):
        script = next((a for a in args[1:] if not a.startswith("-")), None)
        if script:
            words.append(os.path.basename(script))
    text = " ".join(words)
    return text if len(text) <= 48 else text[:47] + "…"


TEFLON_LIB = re.compile(r"/([^/]+)/build/[^ ]*libteflon\.so[^/]*$")


def _read_maps(path, nodes):
    """(bytes of NPU buffers mapped, Mesa tree of the mapped libteflon.so or None)."""
    total = 0
    teflon = None
    try:
        with open(path) as f:
            for line in f:
                parts = line.split(maxsplit=5)
                if len(parts) != 6:
                    continue
                name = parts[5].rstrip("\n")
                if name in nodes:
                    lo, hi = parts[0].split("-")
                    total += int(hi, 16) - int(lo, 16)
                elif teflon is None and "libteflon.so" in name:
                    m = TEFLON_LIB.search(name)
                    teflon = m.group(1) if m else "libteflon.so"
    except (OSError, ValueError):
        pass
    return total, teflon


def _read_environ(path, names):
    """The named variables as the process was started with them. Later changes
    inside the process (os.environ in Python) don't show up here."""
    found = {}
    try:
        with open(path, "rb") as f:
            for item in f.read().split(b"\0"):
                key, sep, value = item.partition(b"=")
                if sep and key.decode(errors="replace") in names:
                    found[key.decode()] = value.decode(errors="replace")[:40]
    except OSError:
        pass
    return found


def read_axi_switch():
    """True/False for the driver's AXI SRAM switch, None if this driver has none."""
    text = read_text(host_path(AXI_SRAM_SWITCH))
    if text is None:
        return None
    return text[:1] in ("Y", "y", "1")


def find_hang_dumps():
    """Names of kernel device dumps waiting to be read. The kernel deletes each
    after about five minutes, so a waiting one means a very recent hang."""
    try:
        return sorted(e for e in os.listdir(host_path(DEVCOREDUMP_DIR)) if e.startswith("devcd"))
    except OSError:
        return []


def stop_file():
    return STOP_FILE if ROOT == "/" else host_path(STOP_FILE)


# --------------------------------------------------------------------------
# Sampling


@dataclass
class Reading:
    t: float
    pm_auto: bool          # runtime power management on (power/control = auto)
    status: str | None     # power/runtime_status
    active_ms: int | None  # power/runtime_active_time
    irqs: int | None
    throttle: int | None
    throttle_max: int | None
    temps: dict


@dataclass
class Status:
    state: str                    # "working", "idle" or "missing"
    npu: Npu | None = None
    pm_auto: bool = True
    active_pct: float | None = None
    irq_rate: float | None = None
    throttle: int | None = None
    throttle_max: int | None = None
    temps: dict = field(default_factory=dict)
    active_total_ms: int | None = None
    irq_total: int | None = None
    clients: list = field(default_factory=list)
    history: list = field(default_factory=list)
    axi_switch: bool | None = None                    # driver's AXI SRAM switch (None: no such switch)
    hang_dumps: list = field(default_factory=list)    # kernel hang dumps waiting to be copied
    stopped: bool = False                             # the kit's runs/STOP exists

    @property
    def alert(self):
        return bool(self.hang_dumps or self.stopped)

    @property
    def clock_pct(self):
        return None if self.throttle is None else 100 / 2 ** self.throttle


def read_irqs(name):
    """Interrupts raised by the device so far, summed over all CPUs."""
    total = None
    try:
        with open(host_path("/proc/interrupts")) as f:
            ncpu = len(f.readline().split())
            for line in f:
                parts = line.split()
                if parts and parts[-1] == name:
                    total = (total or 0) + sum(int(x) for x in parts[1:1 + ncpu] if x.isdigit())
    except OSError:
        pass
    return total


class Monitor:
    """Turns successive readings into Status objects. Reads files only."""

    def __init__(self):
        self.zones = _thermal_zones()
        self.npu = find_npu()
        self.prev = self._read() if self.npu else None
        self.history = deque(maxlen=HISTORY)
        self.clients = []
        self._looked_at = time.monotonic()
        self._scanned_at = -math.inf
        self._last_state = None

    def _temps(self):
        temps = {}
        for label, path in self.zones:
            millideg = read_int(path)
            if millideg is not None:
                temps[label] = millideg / 1000
        return temps

    def _read(self):
        power = os.path.join(self.npu.sysfs, "power")
        cooling = self.npu.cooling
        return Reading(
            t=time.monotonic(),
            pm_auto=read_text(os.path.join(power, "control")) != "on",
            status=read_text(os.path.join(power, "runtime_status")),
            active_ms=read_int(os.path.join(power, "runtime_active_time")),
            irqs=read_irqs(self.npu.name),
            throttle=read_int(os.path.join(cooling, "cur_state")) if cooling else None,
            throttle_max=read_int(os.path.join(cooling, "max_state")) if cooling else None,
            temps=self._temps())

    def update(self):
        """Take a reading; rates cover the time since the previous one."""
        now = time.monotonic()
        if self.npu and not os.path.isdir(self.npu.sysfs):
            self.npu = None  # driver unloaded
        if self.npu is None:
            if now - self._looked_at >= 10:
                self._looked_at = now
                self.npu = find_npu()
                self.prev = None
            if self.npu is None:
                return Status("missing", temps=self._temps(), hang_dumps=find_hang_dumps(),
                              stopped=os.path.exists(stop_file()))

        cur = self._read()
        prev, self.prev = self.prev, cur
        st = Status("idle", npu=self.npu, pm_auto=cur.pm_auto,
                    throttle=cur.throttle, throttle_max=cur.throttle_max, temps=cur.temps,
                    active_total_ms=cur.active_ms, irq_total=cur.irqs,
                    axi_switch=read_axi_switch(), hang_dumps=find_hang_dumps(),
                    stopped=os.path.exists(stop_file()))
        d_active = d_irq = 0
        if prev is not None:
            dt = max(cur.t - prev.t, 1e-3)
            if cur.irqs is not None and prev.irqs is not None:
                d_irq = max(cur.irqs - prev.irqs, 0)
                st.irq_rate = d_irq / dt
            # With runtime PM off the NPU is always powered, so power-on time says nothing.
            if cur.pm_auto and cur.active_ms is not None and prev.active_ms is not None:
                d_active = max(cur.active_ms - prev.active_ms, 0)
                st.active_pct = min(100.0, d_active / (dt * 10))  # ms per s -> %
        powered = cur.pm_auto and cur.status == "active"
        if powered or d_active or d_irq:
            st.state = "working"
        if st.active_pct is not None:
            self.history.append(st.active_pct)
        elif not cur.pm_auto and prev is not None:
            self.history.append(100.0 if d_irq else 0.0)  # kept on: did it raise interrupts?
        st.history = list(self.history)

        # A process scan costs ~20 ms here, so run it every few seconds,
        # and straight away when work starts.
        every = 3 if st.state == "working" else 10
        started = st.state == "working" and self._last_state != "working"
        if started or now - self._scanned_at >= every:
            self.clients = find_clients(self.npu.nodes)
            self._scanned_at = now
        self._last_state = st.state
        st.clients = self.clients
        return st


# --------------------------------------------------------------------------
# Text shared by the terminal and the top-bar menu


def fmt_duration(ms):
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def fmt_bytes(n):
    return f"{n / 2**20:.0f} MiB" if n >= 2**20 else f"{n / 1024:.0f} KiB"


def headline(st):
    if st.state == "missing":
        return "Not found (is the etnaviv driver loaded?)"
    if not st.pm_auto:  # kept powered on: interrupts are the only sign of work
        if st.state == "idle":
            return "Idle · kept powered on"
        if st.irq_rate is None:
            return "Working · kept powered on"
        return f"Working · {st.irq_rate:,.0f} interrupts/s · kept on"
    if st.state == "idle":
        return "Idle · powered down"
    if st.active_pct is None:
        return "Working"
    return f"Working · {st.active_pct:.0f}% active"


def alert_rows(st):
    rows = []
    if st.hang_dumps:
        rows.append(("⚠ Hang dump", f"{len(st.hang_dumps)} waiting in {DEVCOREDUMP_DIR} · "
                                    "copy it first (needs sudo), it is deleted after ~5 min"))
    if st.stopped:
        rows.append(("⚠ NPU jobs", "blocked since a hang (runs/STOP) · reboot before the next NPU job"))
    return rows


def memory_row(st):
    cache = f"{fmt_bytes(ONCHIP_CACHE_BYTES)} cache"
    if st.axi_switch is None:
        return f"{cache} · {fmt_bytes(AXI_SRAM_BYTES)} AXI SRAM (this driver can't use it)"
    if not st.axi_switch:
        return f"{cache} · {fmt_bytes(AXI_SRAM_BYTES)} AXI SRAM (driver switch off)"
    asked = [c for c in st.clients if c.axi]
    use = f"asked for by {len(asked)} program{'s' if len(asked) != 1 else ''}" if asked else "no program asks for it"
    return f"{cache} · {fmt_bytes(AXI_SRAM_BYTES)} AXI SRAM (driver switch on · {use})"


def shader_row(st):
    ops = sorted({op.strip() for c in st.clients if c.shader for op in c.shader.split(",") if op.strip()})
    if not ops:
        return "not switched on by any program"
    n = sum(1 for c in st.clients if c.shader)
    return f"{', '.join(ops)} · switched on in {n} program{'s' if n != 1 else ''}"


def detail_rows(st):
    if st.npu is None:
        return alert_rows(st)
    rows = alert_rows(st)
    rows += [("Interrupts", "—" if st.irq_rate is None else f"{st.irq_rate:,.0f}/s"),
            ("Power", "powers down when idle" if st.pm_auto else "kept on (power/control = on)")]
    if st.throttle is None:
        clock = "unknown"
    elif st.throttle == 0:
        clock = "full speed"
    else:
        clock = (f"throttled to {st.clock_pct:g}% "
                 f"(thermal step {st.throttle} of {st.throttle_max})")
    rows.append(("Clock", clock))
    rows.append(("Memory", memory_row(st)))
    rows.append(("Shader core", shader_row(st)))
    if st.temps:
        rows.append(("Temperature", " · ".join(f"{k} {v:.0f} °C" for k, v in st.temps.items())))
    totals = []
    if st.active_total_ms is not None:
        totals.append(f"powered {fmt_duration(st.active_total_ms)}")
    if st.irq_total is not None:
        totals.append(f"{st.irq_total:,} interrupts")
    if totals:
        rows.append(("Since boot", " · ".join(totals)))
    return rows


def top_bar_label(st):
    """Text beside the top-bar icon, and the widest it can get (keeps the spacing steady)."""
    if st.state != "working":
        return "", ""
    if st.active_pct is not None:
        return f"{st.active_pct:.0f}%", "100%"
    if not st.pm_auto and st.irq_rate is not None:
        return f"{st.irq_rate:,.0f}/s", "1,000/s"
    return "", ""


def device_line(st):
    if st.npu is None:
        return "No etnaviv NPU device found"
    return " · ".join(filter(None, [f"{st.npu.name} (etnaviv)", st.npu.board]))


def program_lines(st):
    if st.clients:
        return [f"{c.cmd or c.name} (pid {c.pid})"
                + (f" · {fmt_bytes(c.mapped)} mapped" if c.mapped else "")
                + (f" · Teflon: {c.teflon}" if c.teflon else "")
                + (f" · shader: {c.shader}" if c.shader else "")
                + (f" · AXI SRAM: {c.axi}" if c.axi else "")
                for c in st.clients]
    if st.state == "working" and os.geteuid() != 0:
        return ["none of yours (another user's process?)"]
    return ["none"]


def as_dict(st):
    pct = st.active_pct
    return {
        "state": st.state,
        "power_control": "auto" if st.pm_auto else "on",
        "active_percent": None if pct is None else round(pct, 1),
        "interrupts_per_second": None if st.irq_rate is None else round(st.irq_rate, 1),
        "clock_percent": st.clock_pct,
        "throttle_state": st.throttle,
        "throttle_max": st.throttle_max,
        "temperatures_c": st.temps,
        "active_total_seconds": None if st.active_total_ms is None else st.active_total_ms / 1000,
        "interrupts_total": st.irq_total,
        "onchip_cache_bytes": ONCHIP_CACHE_BYTES,
        "axi_sram_bytes": AXI_SRAM_BYTES,
        "axi_sram_driver_switch": st.axi_switch,
        "hang_dumps_waiting": st.hang_dumps,
        "npu_jobs_blocked": st.stopped,
        "programs": [{"pid": c.pid, "name": c.name, "command": c.cmd, "mapped_bytes": c.mapped,
                      "teflon": c.teflon, "shader_ops": c.shader, "axi_sram": c.axi}
                     for c in st.clients],
        "device": st.npu.name if st.npu else None,
        "drm_nodes": sorted(st.npu.nodes) if st.npu else [],
        "board": st.npu.board if st.npu else None,
    }


# --------------------------------------------------------------------------
# Terminal


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values):
    return "".join(SPARK[min(len(SPARK) - 1, int(v / 100 * len(SPARK)))] for v in values)


def render(st, color=False, spark_seconds=None):
    def paint(code, text):
        return f"\x1b[{code}m{text}\x1b[0m" if color else text

    tone = {"working": "1;32", "idle": "1", "missing": "1;31"}[st.state]
    if st.alert or (st.state == "working" and st.throttle):
        tone = "1;33"
    lines = [f" {'NPU':<12} {paint(tone, headline(st))}"]
    if spark_seconds is not None and st.history:
        lines.append(f" {'':<12} {paint('32', sparkline(st.history))}"
                     f"  {paint('2', f'last {spark_seconds:.0f} s')}")
    lines += [f" {k:<12} {v}" for k, v in detail_rows(st)]
    lines.append(f" {'Device':<12} {device_line(st)}")
    if st.npu is not None:
        lines += ["", " Programs using the NPU"]
        lines += [f"   {p}" for p in program_lines(st)]
    return "\n".join(lines)


def run_once(interval, as_json):
    monitor = Monitor()
    time.sleep(interval)
    st = monitor.update()
    if as_json:
        print(json.dumps(as_dict(st), indent=2, ensure_ascii=False))
    else:
        print(render(st, color=sys.stdout.isatty()))


def run_watch(interval):
    monitor = Monitor()
    tty = sys.stdout.isatty()
    out = sys.stdout
    if tty:
        out.write("\x1b]0;NPU status\x07\x1b[?25l\x1b[2J")  # title, hide cursor, clear
    try:
        while True:
            time.sleep(interval)
            st = monitor.update()
            text = render(st, color=tty, spark_seconds=len(st.history) * interval)
            if tty:
                hint = "\n\n \x1b[2mCtrl+C to quit\x1b[0m"
                out.write("\x1b[H" + (text + hint).replace("\n", "\x1b[K\n") + "\x1b[K\x1b[J")
            else:
                out.write(text + "\n\n")
            out.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            out.write("\x1b[?25h\n")


# --------------------------------------------------------------------------
# Top-bar icons: a chip whose centre fills as the NPU works


def icon_name(kind, level=0):
    return f"{APP_ID}-{ICON_VERSION}-{kind}{level or ''}-symbolic"


def icon_for(st):
    if st.state == "missing":
        return icon_name("off")
    if st.alert:
        return icon_name("alert")
    if st.state != "working":
        return icon_name("idle")
    pct = 100 if st.active_pct is None else st.active_pct
    level = max(1, min(LEVELS, math.ceil(pct / 100 * LEVELS)))
    return icon_name("hot" if st.throttle else "load", level)


def chip_svg(level=0, accent="success", dim=False):
    """16 px symbolic icon. Shapes are filled, never stroked, so the shell can
    recolour them; the fill colours are a fallback for a dark top bar."""
    fg = "#f2f2f2"
    accents = {"success": "#33d17a", "warning": "#ffa348"}
    body = f'<path fill="{fg}" fill-rule="evenodd" d="M3 3h10v10H3zM5 5v6h6V5z"/>'
    pins = "".join(f'<rect x="{x}" y="{y}" width="2" height="2" fill="{fg}"/>'
                   for c in (5, 9)
                   for x, y in ((c, 1), (c, 13), (1, c), (13, c)))
    core = ""
    if level:
        core = (f'<rect class="{accent}" x="6" y="{10 - level}" width="4" height="{level}" '
                f'fill="{accents[accent]}"/>')
    opacity = ' opacity="0.45"' if dim else ""
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
            f'<g{opacity}>{body}{pins}{core}</g></svg>\n')


def write_icons():
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~/.cache")
    folder = os.path.join(base, APP_ID, "icons")
    os.makedirs(folder, exist_ok=True)
    icons = {icon_name("idle"): chip_svg(), icon_name("off"): chip_svg(dim=True),
             icon_name("alert"): chip_svg(LEVELS, "warning")}
    for level in range(1, LEVELS + 1):
        icons[icon_name("load", level)] = chip_svg(level, "success")
        icons[icon_name("hot", level)] = chip_svg(level, "warning")
    for name, svg in icons.items():
        with open(os.path.join(folder, name + ".svg"), "w") as f:
            f.write(svg)
    return folder


# --------------------------------------------------------------------------
# Top-bar indicator


def config_home():
    return os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")


SETTINGS_FILE = os.path.join(config_home(), APP_ID, "settings.json")


def load_settings():
    settings = {"show_label": True}
    try:
        with open(SETTINGS_FILE) as f:
            settings.update(json.load(f))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)


def single_instance_lock():
    import fcntl
    folder = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    lock = open(os.path.join(folder, f"{APP_ID}.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock  # held until the process exits


def weight_cache_dir():
    """Where Teflon keeps compiled NPU weights; the same lookup as npu_cache_default_dir()."""
    if os.environ.get("ETNA_ML_DISK_CACHE_DIR"):
        return os.environ["ETNA_ML_DISK_CACHE_DIR"]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "teflon-npu")


def read_disk(cache_dir):
    """(free bytes on the disk holding the cache, bytes the cache holds), or None."""
    place = cache_dir
    while not os.path.isdir(place):  # no cache yet: measure the disk it would be made on
        parent = os.path.dirname(place)
        if parent == place:
            return None
        place = parent
    try:
        fs = os.statvfs(place)
        used = 0
        if place == cache_dir:
            with os.scandir(cache_dir) as entries:
                for entry in entries:
                    if entry.name.endswith(".npucache") and entry.is_file(follow_symlinks=False):
                        used += entry.stat(follow_symlinks=False).st_size
    except OSError:
        return None
    return fs.f_bavail * fs.f_frsize, used


def disk_line(free, used):
    gib = free / 1024 ** 3
    if free < DISK_LOW_BYTES:
        return f"⚠ Disk low: {gib:.1f} GiB free · weight cache ({fmt_bytes(used)}) can't grow"
    return f"Weight cache: {fmt_bytes(used)} on disk · {gib:.1f} GiB free"


def send_notification(title, body, icon="drive-harddisk"):
    """Desktop notification. Never raises: a missing notify-send only loses the popup."""
    try:
        subprocess.Popen(["notify-send", "--app-name=NPU status", f"--icon={icon}",
                          "--urgency=critical", title, body],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return True
    except OSError:
        return False


class DiskWatch:
    """Notifies once when free space drops under DISK_LOW_BYTES, and again only after it
    has recovered past DISK_REARM_BYTES, so hovering around the line doesn't nag."""

    def __init__(self, notify=send_notification):
        self.notify = notify
        self.armed = True

    def check(self, free, used):
        if free >= DISK_REARM_BYTES:
            self.armed = True
        elif free < DISK_LOW_BYTES and self.armed:
            self.armed = False
            self.notify("Disk space is low",
                        f"{free / 1024 ** 3:.1f} GiB free, under {DISK_LOW_BYTES // 1024 ** 3} GiB. "
                        f"The NPU weight cache ({fmt_bytes(used)}) won't grow until space is freed.")
            return True
        return False


class HangWatch:
    """One popup per new hang dump, and one when NPU jobs become blocked."""

    def __init__(self, notify=send_notification):
        self.notify = notify
        self.seen = set()
        self.stopped = False

    def check(self, st):
        sent = False
        new = set(st.hang_dumps) - self.seen
        self.seen = set(st.hang_dumps)
        if new:
            sent = True
            self.notify("NPU hang: copy the dump now",
                        f"A kernel dump is waiting ({', '.join(sorted(new))}) and is deleted after about "
                        f"5 minutes. Copy it before anything else: "
                        f"sudo cp {DEVCOREDUMP_DIR}/devcd*/data ~/alta-npu/incidents/",
                        icon="dialog-warning")
        if st.stopped and not self.stopped and not new:
            sent = True
            self.notify("NPU jobs are blocked",
                        "The kit stopped NPU jobs after a hang (runs/STOP). Reboot before the next NPU job.",
                        icon="dialog-warning")
        self.stopped = st.stopped
        return sent


class TrayApp:
    PROGRAM_ROWS = 5

    def __init__(self, Gtk, GLib, AppIndicator, interval):
        self.Gtk, self.GLib = Gtk, GLib
        self.interval = interval
        self.monitor = Monitor()
        self.settings = load_settings()
        self._icon = icon_name("idle")
        self._label = ""

        self.indicator = AppIndicator.Indicator.new_with_path(
            APP_ID, self._icon, AppIndicator.IndicatorCategory.HARDWARE, write_icons())
        self.indicator.set_title("NPU status")
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)

        menu = Gtk.Menu()

        def add(item):
            menu.append(item)
            return item

        self.head = add(Gtk.MenuItem(label="Reading…"))
        self.detail = [add(Gtk.MenuItem(label="")) for _ in range(9)]
        self.hang_watch = HangWatch()
        self.queue = add(Gtk.MenuItem(label=""))
        self.queue.connect("activate", lambda *_: self._open_queue_window())
        self.trip_watch = npu_queue.TripWatch() if npu_queue else None
        self.disk = add(Gtk.MenuItem(label=""))
        self.disk_watch = DiskWatch()
        self._disk_checked = None
        add(Gtk.SeparatorMenuItem())
        self.programs_head = add(Gtk.MenuItem(label="Programs using the NPU:"))
        self.programs = [add(Gtk.MenuItem(label="")) for _ in range(self.PROGRAM_ROWS)]
        add(Gtk.SeparatorMenuItem())
        self.device = add(Gtk.MenuItem(label=""))
        add(Gtk.SeparatorMenuItem())
        toggle = add(Gtk.CheckMenuItem(label="Show load in top bar"))
        toggle.set_active(self.settings["show_label"])
        toggle.connect("toggled", self._on_toggle_label)
        if npu_queue:
            add(Gtk.MenuItem(label="Job queue…")).connect("activate",
                                                         lambda *_: self._open_queue_window())
        if shutil.which("gnome-terminal"):
            add(Gtk.MenuItem(label="Open live view")).connect("activate", self._on_live_view)
        add(Gtk.MenuItem(label="Quit")).connect("activate", lambda *_: Gtk.main_quit())
        menu.show_all()
        for item in self.detail + self.programs + [self.disk, self.queue]:
            item.hide()
        self.indicator.set_menu(menu)

        for sig in (signal.SIGINT, signal.SIGTERM):
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, self._quit)
        GLib.timeout_add(int(interval * 1000), self._tick)
        if npu_queue:
            GLib.idle_add(self._queue_at_login)

    def _quit(self):
        self.Gtk.main_quit()
        return self.GLib.SOURCE_REMOVE

    @staticmethod
    def _show(item, text):
        """Update a row only when it changes, so an idle NPU costs no D-Bus traffic."""
        if text is None:
            if item.get_visible():
                item.hide()
            return
        if item.get_label() != text:
            item.set_label(text)
        if not item.get_visible():
            item.show()

    def _tick(self):
        try:
            self._apply(self.monitor.update())
        except Exception:  # one bad reading must not take the indicator down
            traceback.print_exc()
        try:
            self._check_disk()
        except Exception:
            traceback.print_exc()
        try:
            self._check_queue()
        except Exception:
            traceback.print_exc()
        return True

    # -- job queue (npu_queue.py, npu_control.py) --------------------------

    def _queue_at_login(self):
        """Apply the saved job-queue settings once per boot; say so if some were held back."""
        try:
            for msg in npu_queue.apply_at_login():
                send_notification("NPU job queue", msg, icon="dialog-information")
        except Exception:
            traceback.print_exc()
        return False

    def _check_queue(self):
        if not npu_queue:
            return
        st = npu_queue.read_state(clients=False)
        self._show(self.queue, npu_queue.summary(st))
        trip = self.trip_watch.check(st)
        if trip:
            frm, n = trip
            send_notification("NPU queue dropped to 1 job",
                              f"The NPU hung and the driver recovered it, so the queue went from "
                              f"{frm} to 1 job at a time ({n}× this boot). Choose in the Job queue "
                              f"window: put it back, keep 1 for now, or reboot.",
                              icon="dialog-warning")
            self._open_queue_window(trip=True)

    def _open_queue_window(self, trip=False):
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "npu_control.py")]
        if trip:
            cmd.append("--trip")
        if ROOT != "/":
            cmd += ["--root", ROOT]
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, start_new_session=True)

    def _check_disk(self):
        now = time.monotonic()  # uptime, not wall time: this board's clock can jump
        if self._disk_checked is not None and now - self._disk_checked < DISK_CHECK_SECONDS:
            return
        self._disk_checked = now
        reading = read_disk(weight_cache_dir())
        if reading is None:
            self._show(self.disk, None)
            return
        self._show(self.disk, disk_line(*reading))
        self.disk_watch.check(*reading)

    def _apply(self, st):
        self.hang_watch.check(st)
        self._show(self.head, headline(st))
        rows = [f"{k}: {v}" for k, v in detail_rows(st)]
        for i, item in enumerate(self.detail):
            self._show(item, rows[i] if i < len(rows) else None)

        lines = program_lines(st) if st.npu else []
        if len(lines) > self.PROGRAM_ROWS:
            extra = len(lines) - self.PROGRAM_ROWS + 1
            lines = lines[:self.PROGRAM_ROWS - 1] + [f"…and {extra} more"]
        self._show(self.programs_head, "Programs using the NPU:" if st.npu else None)
        for i, item in enumerate(self.programs):
            self._show(item, "    " + lines[i] if i < len(lines) else None)
        self._show(self.device, device_line(st))

        icon = icon_for(st)
        if icon != self._icon:
            self.indicator.set_icon_full(icon, headline(st))
            self._icon = icon
        label, guide = top_bar_label(st) if self.settings["show_label"] else ("", "")
        if label != self._label:
            self.indicator.set_label(label, guide)
            self._label = label

    def _on_toggle_label(self, item):
        self.settings["show_label"] = item.get_active()
        save_settings(self.settings)
        if not item.get_active() and self._label:
            self.indicator.set_label("", "")
            self._label = ""

    def _on_live_view(self, _item):
        cmd = [sys.executable, os.path.abspath(__file__), "--watch", "--interval", str(self.interval)]
        if ROOT != "/":
            cmd += ["--root", ROOT]
        subprocess.Popen(["gnome-terminal", "--"] + cmd, start_new_session=True)


def run_tray(interval):
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        gi.require_version("AyatanaAppIndicator3", "0.1")
        from gi.repository import AyatanaAppIndicator3, GLib, Gtk
    except (ImportError, ValueError) as e:
        sys.exit(f"npu-status: the top-bar indicator needs PyGObject and AyatanaAppIndicator3 "
                 f"({e}).\nRun it with /usr/bin/python3, or use --watch for the terminal view.")
    if not Gtk.init_check(sys.argv[:1])[0]:
        sys.exit("npu-status: no graphical session found. Use --watch for the terminal view.")
    lock = single_instance_lock()
    if lock is None:
        print("npu-status: the indicator is already running.")
        return
    TrayApp(Gtk, GLib, AyatanaAppIndicator3, interval)
    Gtk.main()


# --------------------------------------------------------------------------
# Autostart


DESKTOP_ENTRY = """\
[Desktop Entry]
Type=Application
Name=NPU Status
Comment=Shows what the NPU is doing in the top bar
Exec={exec}
Icon=utilities-system-monitor
Terminal=false
Categories=System;Monitor;
X-GNOME-Autostart-enabled=true
"""


def _desktop_quote(arg):
    if not any(ch in arg for ch in " \t\n\"'\\><~|&;$*?#()`"):
        return arg
    return '"' + "".join("\\" + ch if ch in '"`$\\' else ch for ch in arg) + '"'


def install(enable):
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    paths = [os.path.join(config_home(), "autostart", f"{APP_ID}.desktop"),
             os.path.join(data_home, "applications", f"{APP_ID}.desktop")]
    if not enable:
        for path in paths:
            try:
                os.remove(path)
                print(f"removed {path}")
            except FileNotFoundError:
                pass
        return
    python = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
    exec_line = " ".join(_desktop_quote(a) for a in (python, os.path.abspath(__file__)))
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(DESKTOP_ENTRY.format(exec=exec_line))
        print(f"wrote {path}")
    print("The indicator now starts at login and is listed in the app menu as 'NPU Status'.")


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Read-only status indicator for the A311D NPU. "
                    "With no options it shows an icon in the top bar.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true", help="live view in the terminal")
    mode.add_argument("--once", action="store_true", help="print one report and exit")
    mode.add_argument("--install", action="store_true",
                      help="start the indicator at login and list it in the app menu")
    mode.add_argument("--uninstall", action="store_true", help="undo --install")
    parser.add_argument("--json", action="store_true", help="with --once: print JSON")
    parser.add_argument("--interval", type=float, default=1.0, metavar="SECONDS",
                        help="sampling interval (default: 1)")
    parser.add_argument("--root", default="/", help=argparse.SUPPRESS)  # fake /sys + /proc, for tests
    args = parser.parse_args()
    if args.json and not args.once:
        parser.error("--json only works with --once")

    global ROOT
    ROOT = os.path.abspath(args.root)
    if npu_queue:
        npu_queue.ROOT = ROOT
    interval = max(args.interval, 0.25)

    if args.once:
        run_once(interval, args.json)
    elif args.watch:
        run_watch(interval)
    elif args.install or args.uninstall:
        install(args.install)
    else:
        run_tray(interval)


if __name__ == "__main__":
    main()
