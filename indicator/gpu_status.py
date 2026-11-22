#!/usr/bin/python3
"""gpu-status: a read-only status indicator for the Amlogic A311D GPU (Mali-G52).

Shows how hard the GPU is working and which programs are using it: as an icon
in the GNOME top bar, as a live terminal view, or as a one-off report.

It only reads kernel status files. It never opens the GPU device or sends it
work. It shares its small helpers with npu_status.py, which must stay beside it.

    gpu_status.py              top-bar indicator
    gpu_status.py --watch      live view in the terminal
    gpu_status.py --once       one report (add --json for JSON)
    gpu_status.py --install    start the indicator at login (--uninstall)

Where the numbers come from:

    busy %           drm-engine-* in /proc/<pid>/fdinfo, added up over programs.
                     The headline is the busier of the two engines (fragment and
                     vertex/tiler). The kernel only counts this while
                     <device>/profiling is 1, which needs root and resets at
                     boot (alta-npu root step 130). The driver warns the figures
                     are approximate. Other users' programs are not visible.
    clock            devfreq trans_stat, averaged over the sample and counting
                     only powered-up time. It is not a utilization figure: the
                     governor goes to full clock once the GPU is ~45% busy.
    job interrupts/s the panfrost-job line in /proc/interrupts.
    powered %        power/runtime_active_time. The GPU powers down 50 ms after
                     its last job, so this runs far above the real load.
    throttle         the GPU's devfreq cooling device.
    temperature      /sys/class/thermal/thermal_zone*. There is no GPU sensor.
    programs         /proc/<pid>/fd entries pointing at the GPU's DRM nodes, and
                     drm-resident-memory from fdinfo. A buffer shared between
                     programs counts once in each of them.
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

import npu_status as shared
from npu_status import (_cooling_device, _describe_cmdline, _desktop_quote, _readlink, _thermal_zones,
                        config_home, fmt_bytes, fmt_duration, host_path, read_int, read_irqs, read_text,
                        sparkline)

APP_ID = "gpu-status"
ICON_VERSION = "v1"  # bump when the icon design changes: the shell caches icons by name
LEVELS = 4           # fill steps in the top-bar icon
HISTORY = 60         # samples kept for the terminal sparkline
SCAN_EVERY = 5       # seconds between /proc scans for programs that have the GPU open
JOB_IRQ = "panfrost-job"
L2_CACHE_BYTES = 128 * 1024  # decoded from the kernel's boot line "Features: L2:0x07110206"
ENGINE_LABELS = {"fragment": "fragment", "vertex-tiler": "vertex/tiler"}
UNITS = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}


# --------------------------------------------------------------------------
# Finding the GPU


@dataclass
class Gpu:
    name: str             # kernel device name, e.g. "ffe40000.gpu"
    sysfs: str            # its sysfs directory
    devfreq: str          # its clock-scaling directory
    nodes: frozenset      # its DRM device nodes, e.g. {"/dev/dri/renderD129"}
    cooling: str | None   # its thermal cooling device directory
    clocks: list          # clock steps in Hz, lowest first
    board: str            # board model from the device tree


def find_gpu():
    """Return the GPU bound to the panfrost driver, or None."""
    drv = host_path("/sys/bus/platform/drivers/panfrost")
    try:
        names = sorted(e for e in os.listdir(drv)
                       if e != "module" and os.path.isdir(os.path.join(drv, e, "power")))
    except OSError:
        return None
    if not names:
        return None
    name = names[0]
    sysfs = os.path.realpath(os.path.join(drv, name))
    devfreq = os.path.join(sysfs, "devfreq", name)
    try:
        nodes = frozenset("/dev/dri/" + n for n in os.listdir(os.path.join(sysfs, "drm"))
                          if n.startswith(("card", "renderD")))
    except OSError:
        nodes = frozenset()
    steps = (read_text(os.path.join(devfreq, "available_frequencies")) or "").split()
    board = read_text(host_path("/proc/device-tree/model")) or ""
    return Gpu(name=name, sysfs=sysfs, devfreq=devfreq, nodes=nodes,
               cooling=_cooling_device(f"devfreq-{name}"),
               clocks=sorted(int(s) for s in steps if s.isdigit()),
               board=board.strip("\0 "))


# --------------------------------------------------------------------------
# Programs that have the GPU open


@dataclass
class Holder:
    pid: int
    fds: list  # its file descriptors open on the GPU
    name: str
    cmd: str


@dataclass
class Program:
    pid: int
    name: str
    cmd: str
    resident: int = 0   # bytes of GPU buffers actually in RAM
    allocated: int = 0  # bytes of GPU buffers, including ones not backed yet
    engine_ns: dict = field(default_factory=dict)  # engine -> GPU time since the last reading
    busy_pct: float | None = None


def find_holders(nodes):
    """Processes holding one of the GPU's device nodes open (that we may inspect)."""
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
            with os.scandir(os.path.join(base, "fd")) as entries:
                fds = [e.name for e in entries if _readlink(e.path) in nodes]
        except OSError:  # gone, or another user's process
            continue
        if fds:
            found.append(Holder(pid=int(pid), fds=fds,
                                name=read_text(os.path.join(base, "comm")) or "?",
                                cmd=_describe_cmdline(os.path.join(base, "cmdline"))))
    return sorted(found, key=lambda h: h.pid)


def read_fdinfo(pid, fd):
    """The numeric drm-* fields of one open GPU file; None once it has closed."""
    fields = {}
    try:
        with open(host_path(f"/proc/{pid}/fdinfo/{fd}")) as f:
            for line in f:
                key, _, value = line.partition(":")
                words = value.split()
                if key.startswith("drm-") and words and words[0].isdigit():
                    fields[key] = int(words[0]) * UNITS.get(words[-1], 1)
    except OSError:
        return None
    return fields or None


# --------------------------------------------------------------------------
# Sampling


@dataclass
class Reading:
    t: float
    pm_auto: bool          # runtime power management on (power/control = auto)
    status: str | None     # power/runtime_status
    active_ms: int | None  # power/runtime_active_time
    irqs: int | None
    clock_hz: int | None   # devfreq cur_freq
    clock_ms: dict         # clock in Hz -> ms spent there while powered
    throttle: int | None
    throttle_max: int | None
    profiling: bool
    temps: dict


@dataclass
class Status:
    state: str                    # "working", "idle" or "missing"
    gpu: Gpu | None = None
    pm_auto: bool = True
    powered: bool = False         # powered up right now
    powered_pct: float | None = None
    powered_total_ms: int | None = None
    profiling: bool = False
    busy_pct: float | None = None  # None while the kernel isn't counting
    engines: dict = field(default_factory=dict)  # engine -> busy %
    clock_hz: float | None = None
    irq_rate: float | None = None
    irq_total: int | None = None
    throttle: int | None = None
    throttle_max: int | None = None
    temps: dict = field(default_factory=dict)
    programs: list = field(default_factory=list)
    history: list = field(default_factory=list)


def read_trans_stat(path):
    """{clock in Hz: total ms at that clock}, from a devfreq trans_stat file."""
    times = {}
    for line in (read_text(path) or "").splitlines():
        m = re.match(r"\*?\s*(\d+):.*?(\d+)\s*$", line)
        if m:
            times[int(m.group(1))] = int(m.group(2))
    return times


def average_clock(before, after):
    """Time-weighted average clock between two trans_stat readings; None if no time passed."""
    spent = {hz: max(ms - before.get(hz, 0), 0) for hz, ms in after.items()}
    total = sum(spent.values())
    return sum(hz * ms for hz, ms in spent.items()) / total if total else None


def load_pct(st):
    """What the icon shows: measured busy %, otherwise clock speed as a share of the maximum."""
    if st.busy_pct is not None:
        return st.busy_pct
    if st.clock_hz and st.gpu and st.gpu.clocks:
        return 100 * st.clock_hz / st.gpu.clocks[-1]
    return 100.0


class Monitor:
    """Turns successive readings into Status objects. Reads files only."""

    def __init__(self):
        self.zones = _thermal_zones()
        self.history = deque(maxlen=HISTORY)
        self.holders = []
        self.engines = {}        # (client id, engine) -> ns, from the last reading
        self.engines_at = None
        self.prev = None
        self._looked_at = time.monotonic()
        self._scanned_at = -math.inf
        self.gpu = find_gpu()
        if self.gpu:
            self.prev = self._read()
            self._read_programs()

    def _temps(self):
        temps = {}
        for label, path in self.zones:
            millideg = read_int(path)
            if millideg is not None:
                temps[label] = millideg / 1000
        return temps

    def _read(self):
        power = os.path.join(self.gpu.sysfs, "power")
        cooling = self.gpu.cooling
        return Reading(
            t=time.monotonic(),
            pm_auto=read_text(os.path.join(power, "control")) != "on",
            status=read_text(os.path.join(power, "runtime_status")),
            active_ms=read_int(os.path.join(power, "runtime_active_time")),
            irqs=read_irqs(JOB_IRQ),
            clock_hz=read_int(os.path.join(self.gpu.devfreq, "cur_freq")),
            clock_ms=read_trans_stat(os.path.join(self.gpu.devfreq, "trans_stat")),
            throttle=read_int(os.path.join(cooling, "cur_state")) if cooling else None,
            throttle_max=read_int(os.path.join(cooling, "max_state")) if cooling else None,
            profiling=read_text(os.path.join(self.gpu.sysfs, "profiling")) == "1",
            temps=self._temps())

    def _read_programs(self):
        """Per-program memory and busy share, plus each engine's total busy share."""
        now = time.monotonic()
        if now - self._scanned_at >= SCAN_EVERY:  # a /proc scan costs ~20 ms here
            self.holders = find_holders(self.gpu.nodes)
            self._scanned_at = now
        elapsed_ns = None if self.engines_at is None else max(now - self.engines_at, 1e-3) * 1e9
        clients, engines, programs = set(), {}, []
        for h in self.holders:
            files = [f for f in (read_fdinfo(h.pid, fd) for fd in h.fds) if f]
            if not files:
                continue  # exited since the last scan
            p = Program(h.pid, h.name, h.cmd)
            for fields in files:
                client = fields.get("drm-client-id")
                if client is None or client in clients:
                    continue  # the same open file, inherited or duplicated
                clients.add(client)
                p.resident += fields.get("drm-resident-memory", 0)
                p.allocated += fields.get("drm-total-memory", 0)
                for key, ns in fields.items():
                    if key.startswith("drm-engine-"):
                        engine = key.removeprefix("drm-engine-")
                        engines[client, engine] = ns
                        spent = max(ns - self.engines.get((client, engine), ns), 0)
                        p.engine_ns[engine] = p.engine_ns.get(engine, 0) + spent
            programs.append(p)
        self.engines, self.engines_at = engines, now

        totals = {}
        for p in programs:
            for engine, ns in p.engine_ns.items():
                totals[engine] = totals.get(engine, 0) + ns
            if elapsed_ns and p.engine_ns:
                p.busy_pct = min(100.0, max(p.engine_ns.values()) / elapsed_ns * 100)
        shares = {e: min(100.0, ns / elapsed_ns * 100) for e, ns in totals.items()} if elapsed_ns else {}
        programs.sort(key=lambda p: (-(p.busy_pct or 0), -p.resident, p.pid))
        return programs, shares

    def update(self):
        """Take a reading; rates cover the time since the previous one."""
        now = time.monotonic()
        if self.gpu and not os.path.isdir(self.gpu.sysfs):
            self.gpu = None  # driver unloaded
        if self.gpu is None:
            if now - self._looked_at >= 10:
                self._looked_at = now
                self.gpu = find_gpu()
                self.prev = None
                self.engines, self.engines_at = {}, None
            if self.gpu is None:
                return Status("missing", temps=self._temps())

        cur = self._read()
        prev, self.prev = self.prev, cur
        programs, engines = self._read_programs()
        st = Status("idle", gpu=self.gpu, pm_auto=cur.pm_auto, powered=cur.status == "active",
                    powered_total_ms=cur.active_ms, profiling=cur.profiling, irq_total=cur.irqs,
                    throttle=cur.throttle, throttle_max=cur.throttle_max, temps=cur.temps,
                    programs=programs)
        d_irq = 0
        if prev is not None:
            dt = max(cur.t - prev.t, 1e-3)
            if cur.irqs is not None and prev.irqs is not None:
                d_irq = max(cur.irqs - prev.irqs, 0)
                st.irq_rate = d_irq / dt
            if cur.pm_auto and cur.active_ms is not None and prev.active_ms is not None:
                st.powered_pct = min(100.0, max(cur.active_ms - prev.active_ms, 0) / (dt * 10))
            st.clock_hz = average_clock(prev.clock_ms, cur.clock_ms)
        if st.clock_hz is None and st.powered:
            st.clock_hz = cur.clock_hz
        if d_irq:
            st.state = "working"
        if cur.profiling:
            st.engines = engines
            st.busy_pct = max(engines.values(), default=0.0)
        if prev is not None:
            self.history.append(load_pct(st) if st.state == "working" else 0.0)
        st.history = list(self.history)
        return st


# --------------------------------------------------------------------------
# Text shared by the terminal and the top-bar menu


def fmt_mhz(hz):
    return f"{hz / 1e6:.0f} MHz"


def headline(st):
    if st.state == "missing":
        return "Not found (is the panfrost driver loaded?)"
    if st.state == "idle":
        return "Idle" if st.powered else "Idle · powered down"
    parts = ["Working"]
    if st.busy_pct is not None:
        parts.append(f"{st.busy_pct:.0f}% busy")
    if st.clock_hz:
        parts.append(fmt_mhz(st.clock_hz))
    return " · ".join(parts)


def detail_rows(st):
    if st.gpu is None:
        return []
    if st.profiling:
        busy = " · ".join(f"{ENGINE_LABELS.get(k, k)} {v:.0f}%" for k, v in sorted(st.engines.items()))
    else:
        busy = "not measured, kernel profiling is off (root step 130)"
    rows = [("Busy", busy or "no data yet")]

    clock = "—"
    if st.clock_hz:
        clock = fmt_mhz(st.clock_hz) + (f" of {fmt_mhz(st.gpu.clocks[-1])}" if st.gpu.clocks else "")
    if st.throttle:
        cap = st.gpu.clocks[-1 - st.throttle] if st.throttle < len(st.gpu.clocks) else None
        limit = f"capped at {fmt_mhz(cap)}" if cap else "capped"
        clock += f" · {limit} by temperature (step {st.throttle} of {st.throttle_max})"
    rows.append(("Clock", clock))
    rows.append(("Job interrupts", "—" if st.irq_rate is None else f"{st.irq_rate:,.0f}/s"))
    if st.pm_auto:
        rows.append(("Powered", "—" if st.powered_pct is None else f"{st.powered_pct:.0f}% of the time"))
    else:
        rows.append(("Powered", "kept on (power/control = on)"))
    rows.append(("Cache", f"{fmt_bytes(L2_CACHE_BYTES)} L2 · usage isn't reported"))
    if st.temps:
        rows.append(("Temperature", " · ".join(f"{k} {v:.0f} °C" for k, v in st.temps.items())))
    totals = []
    if st.powered_total_ms is not None:
        totals.append(f"powered {fmt_duration(st.powered_total_ms)}")
    if st.irq_total is not None:
        totals.append(f"{st.irq_total:,} job interrupts")
    if totals:
        rows.append(("Since boot", " · ".join(totals)))
    return rows


def top_bar_label(st):
    """Text beside the top-bar icon, and the widest it can get (keeps the spacing steady).

    Rounded, so the top bar isn't redrawn every sample by the GPU it is measuring."""
    if st.state != "working":
        return "", ""
    if st.busy_pct is not None:
        return f"{5 * round(st.busy_pct / 5)}%", "100%"
    if st.clock_hz:
        return f"{50 * round(st.clock_hz / 50e6)} MHz", "800 MHz"
    return "", ""


def device_line(st):
    if st.gpu is None:
        return "No panfrost GPU device found"
    return " · ".join(filter(None, [f"{st.gpu.name} (panfrost)", st.gpu.board]))


def program_lines(st):
    using = [p for p in st.programs if p.resident or p.allocated or p.busy_pct]
    lines = []
    for p in using:
        parts = [f"{p.cmd or p.name} (pid {p.pid})"]
        if p.busy_pct is not None:
            parts.append(f"{p.busy_pct:.0f}%")
        parts.append(fmt_bytes(p.resident))
        lines.append(" · ".join(parts))
    idle = len(st.programs) - len(using)
    if idle:
        lines.append(f"{idle} more with the GPU open, nothing allocated")
    return lines or ["none"]


def as_dict(st):
    def rounded(value, digits=1):
        return None if value is None else round(value, digits)

    return {
        "state": st.state,
        "profiling": st.profiling,
        "busy_percent": rounded(st.busy_pct),
        "engines_percent": {k: round(v, 1) for k, v in st.engines.items()},
        "clock_mhz": None if st.clock_hz is None else round(st.clock_hz / 1e6),
        "clock_max_mhz": round(st.gpu.clocks[-1] / 1e6) if st.gpu and st.gpu.clocks else None,
        "job_interrupts_per_second": rounded(st.irq_rate),
        "job_interrupts_total": st.irq_total,
        "power_control": "auto" if st.pm_auto else "on",
        "powered_percent": rounded(st.powered_pct),
        "powered_total_seconds": None if st.powered_total_ms is None else st.powered_total_ms / 1000,
        "throttle_state": st.throttle,
        "throttle_max": st.throttle_max,
        "temperatures_c": st.temps,
        "l2_cache_bytes": L2_CACHE_BYTES,
        "programs": [{"pid": p.pid, "name": p.name, "command": p.cmd,
                      "busy_percent": rounded(p.busy_pct),
                      "resident_bytes": p.resident, "allocated_bytes": p.allocated}
                     for p in st.programs],
        "device": st.gpu.name if st.gpu else None,
        "drm_nodes": sorted(st.gpu.nodes) if st.gpu else [],
        "board": st.gpu.board if st.gpu else None,
    }


# --------------------------------------------------------------------------
# Terminal


def render(st, color=False, spark_seconds=None):
    def paint(code, text):
        return f"\x1b[{code}m{text}\x1b[0m" if color else text

    tone = {"working": "1;32", "idle": "1", "missing": "1;31"}[st.state]
    if st.state == "working" and st.throttle:
        tone = "1;33"
    lines = [f" {'GPU':<15} {paint(tone, headline(st))}"]
    if spark_seconds is not None and st.history:
        lines.append(f" {'':<15} {paint('32', sparkline(st.history))}"
                     f"  {paint('2', f'last {spark_seconds:.0f} s')}")
    lines += [f" {k:<15} {v}" for k, v in detail_rows(st)]
    lines.append(f" {'Device':<15} {device_line(st)}")
    if st.gpu is not None:
        lines += ["", " Programs using the GPU"]
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
        out.write("\x1b]0;GPU status\x07\x1b[?25l\x1b[2J")  # title, hide cursor, clear
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
# Top-bar icons: a graphics card whose window fills as the GPU works


def icon_name(kind, level=0):
    return f"{APP_ID}-{ICON_VERSION}-{kind}{level or ''}-symbolic"


def icon_for(st):
    if st.state == "missing":
        return icon_name("off")
    if st.state != "working":
        return icon_name("idle")
    level = max(1, min(LEVELS, math.ceil(load_pct(st) / 100 * LEVELS)))
    return icon_name("hot" if st.throttle else "load", level)


def card_svg(level=0, accent="success", dim=False):
    """16 px symbolic icon. Shapes are filled, never stroked, so the shell can
    recolour them; the fill colours are a fallback for a dark top bar."""
    fg = "#f2f2f2"
    accents = {"success": "#33d17a", "warning": "#ffa348"}
    bracket = f'<rect x="1" y="2" width="1" height="12" fill="{fg}"/>'
    body = f'<path fill="{fg}" fill-rule="evenodd" d="M3 3h12v8H3zM5 4v6h8V4z"/>'
    fingers = "".join(f'<rect x="{x}" y="11" width="2" height="2" fill="{fg}"/>' for x in (5, 8, 11))
    core = ""
    if level:
        core = (f'<rect class="{accent}" x="6" y="{9 - level}" width="6" height="{level}" '
                f'fill="{accents[accent]}"/>')
    opacity = ' opacity="0.45"' if dim else ""
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
            f'<g{opacity}>{bracket}{body}{fingers}{core}</g></svg>\n')


def write_icons():
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~/.cache")
    folder = os.path.join(base, APP_ID, "icons")
    os.makedirs(folder, exist_ok=True)
    icons = {icon_name("idle"): card_svg(), icon_name("off"): card_svg(dim=True)}
    for level in range(1, LEVELS + 1):
        icons[icon_name("load", level)] = card_svg(level, "success")
        icons[icon_name("hot", level)] = card_svg(level, "warning")
    for name, svg in icons.items():
        with open(os.path.join(folder, name + ".svg"), "w") as f:
            f.write(svg)
    return folder


# --------------------------------------------------------------------------
# Top-bar indicator


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


class TrayApp:
    PROGRAM_ROWS = 8

    def __init__(self, Gtk, GLib, AppIndicator, interval):
        self.Gtk, self.GLib = Gtk, GLib
        self.interval = interval
        self.monitor = Monitor()
        self.settings = load_settings()
        self._icon = icon_name("idle")
        self._label = ""

        self.indicator = AppIndicator.Indicator.new_with_path(
            APP_ID, self._icon, AppIndicator.IndicatorCategory.HARDWARE, write_icons())
        self.indicator.set_title("GPU status")
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)

        menu = Gtk.Menu()

        def add(item):
            menu.append(item)
            return item

        self.head = add(Gtk.MenuItem(label="Reading…"))
        self.detail = [add(Gtk.MenuItem(label="")) for _ in range(7)]
        add(Gtk.SeparatorMenuItem())
        self.programs_head = add(Gtk.MenuItem(label="Programs using the GPU:"))
        self.programs = [add(Gtk.MenuItem(label="")) for _ in range(self.PROGRAM_ROWS)]
        add(Gtk.SeparatorMenuItem())
        self.device = add(Gtk.MenuItem(label=""))
        add(Gtk.SeparatorMenuItem())
        toggle = add(Gtk.CheckMenuItem(label="Show load in top bar"))
        toggle.set_active(self.settings["show_label"])
        toggle.connect("toggled", self._on_toggle_label)
        if shutil.which("gnome-terminal"):
            add(Gtk.MenuItem(label="Open live view")).connect("activate", self._on_live_view)
        add(Gtk.MenuItem(label="Quit")).connect("activate", lambda *_: Gtk.main_quit())
        menu.show_all()
        for item in self.detail + self.programs:
            item.hide()
        self.indicator.set_menu(menu)

        for sig in (signal.SIGINT, signal.SIGTERM):
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, self._quit)
        GLib.timeout_add(int(interval * 1000), self._tick)

    def _quit(self):
        self.Gtk.main_quit()
        return self.GLib.SOURCE_REMOVE

    @staticmethod
    def _show(item, text):
        """Update a row only when it changes, to keep D-Bus traffic down."""
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
        return True

    def _apply(self, st):
        self._show(self.head, headline(st))
        rows = [f"{k}: {v}" for k, v in detail_rows(st)]
        for i, item in enumerate(self.detail):
            self._show(item, rows[i] if i < len(rows) else None)

        lines = program_lines(st) if st.gpu else []
        if len(lines) > self.PROGRAM_ROWS:
            extra = len(lines) - self.PROGRAM_ROWS + 1
            lines = lines[:self.PROGRAM_ROWS - 1] + [f"…and {extra} more"]
        self._show(self.programs_head, "Programs using the GPU:" if st.gpu else None)
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
        cmd = [sys.executable, os.path.abspath(__file__), "--watch", "--interval", "1"]
        if shared.ROOT != "/":
            cmd += ["--root", shared.ROOT]
        subprocess.Popen(["gnome-terminal", "--"] + cmd, start_new_session=True)


def run_tray(interval):
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        gi.require_version("AyatanaAppIndicator3", "0.1")
        from gi.repository import AyatanaAppIndicator3, GLib, Gtk
    except (ImportError, ValueError) as e:
        sys.exit(f"gpu-status: the top-bar indicator needs PyGObject and AyatanaAppIndicator3 "
                 f"({e}).\nRun it with /usr/bin/python3, or use --watch for the terminal view.")
    if not Gtk.init_check(sys.argv[:1])[0]:
        sys.exit("gpu-status: no graphical session found. Use --watch for the terminal view.")
    lock = single_instance_lock()
    if lock is None:
        # Clicked in the app grid while the indicator already runs (it starts at login): show something
        # instead of exiting silently - the live view, this app's only window.
        if shutil.which("gnome-terminal"):
            print("gpu-status: the indicator is already running; opening the live view.")
            cmd = [sys.executable, os.path.abspath(__file__), "--watch", "--interval", "1"]
            if shared.ROOT != "/":
                cmd += ["--root", shared.ROOT]
            subprocess.Popen(["gnome-terminal", "--"] + cmd, stdin=subprocess.DEVNULL, start_new_session=True)
        else:
            print("gpu-status: the indicator is already running (use its top-bar menu).")
        return
    TrayApp(Gtk, GLib, AyatanaAppIndicator3, interval)
    Gtk.main()


# --------------------------------------------------------------------------
# Autostart


DESKTOP_ENTRY = """\
[Desktop Entry]
Type=Application
Name=GPU Status
Comment=Shows what the GPU is doing in the top bar
Exec={exec}
Icon=utilities-system-monitor
Terminal=false
Categories=System;Monitor;
X-GNOME-Autostart-enabled=true
"""


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
    print("The indicator now starts at login and is listed in the app menu as 'GPU Status'.")


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Read-only status indicator for the A311D GPU. "
                    "With no options it shows an icon in the top bar.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true", help="live view in the terminal")
    mode.add_argument("--once", action="store_true", help="print one report and exit")
    mode.add_argument("--install", action="store_true",
                      help="start the indicator at login and list it in the app menu")
    mode.add_argument("--uninstall", action="store_true", help="undo --install")
    parser.add_argument("--json", action="store_true", help="with --once: print JSON")
    parser.add_argument("--interval", type=float, metavar="SECONDS",
                        help="sampling interval (default: 2 in the top bar, otherwise 1)")
    parser.add_argument("--root", default="/", help=argparse.SUPPRESS)  # fake /sys + /proc, for tests
    args = parser.parse_args()
    if args.json and not args.once:
        parser.error("--json only works with --once")

    shared.ROOT = os.path.abspath(args.root)
    interval = args.interval if args.interval is not None else (1.0 if args.once or args.watch else 2.0)
    interval = max(interval, 0.25)

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
