"""npu_queue: read and change the NPU job-queue settings of the alta etnaviv driver.

Shared by npu_status.py (top-bar indicator) and npu_control.py (the settings window).
Everything here is plain file reads and writes under /sys/module/etnaviv/parameters;
nothing opens the NPU device or sends it work.

Driver parameters (alta kernel patches 0006 and 0007; a stock driver has only hw_job_limit,
read-only):

    hw_job_limit          jobs in the NPU's hardware queue at once, 1-4
    hw_job_auto           Y: a recovered hang drops hw_job_limit to 1
    hw_job_trips          how often that happened since the driver loaded
    hw_job_trip_from      hw_job_limit before the last trip (0: none)
    hw_job_turns          Y: jobs from different programs never share the queue
    hw_job_prio_guard_ms  after a higher-priority job, lower ones enter only an empty queue
    hw_job_prio           priority by process name: name=high|normal|low,...
    hw_job_clients        programs with the NPU open: pid name priority applied|pending

Saved settings (by program name, applied once per boot when the indicator starts) live in
~/.config/npu-status/queue.json; which trip the user has already answered, and the boot the
settings were last applied in, in ~/.config/npu-status/queue-state.json.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field

ROOT = "/"   # tests point this at a fake /sys + /proc tree
PARAM_DIR = "/sys/module/etnaviv/parameters"
BOOT_ID = "/proc/sys/kernel/random/boot_id"
# The alta-npu kit's npu-run writes this after an NPU hang and refuses NPU jobs until a reboot.
STOP_FILE = os.path.join(os.path.expanduser("~"), "alta-npu", "runs", "STOP")

LEVELS = ("high", "normal", "low")
MAX_NAMES = 16       # ETNA_PRIO_NAMES in the driver
NAME_MAX = 15        # TASK_COMM_LEN - 1: the kernel matches /proc/<pid>/comm
LIMIT_MIN, LIMIT_MAX = 1, 4
GUARD_MAX_MS = 1000

DEFAULT_CONFIG = {
    "priorities": {},       # name -> "high" | "normal" | "low"
    "limit": None,          # None: leave the boot value (modprobe.d)
    "auto": None,           # None: leave the driver default
    "turns": None,
    "guard_ms": None,
    "apply_at_login": True,
}

# Writable parameters, as the udev rule of root step 250 lists them.
WRITABLE = ("hw_job_limit", "hw_job_auto", "hw_job_turns", "hw_job_prio_guard_ms", "hw_job_prio")


def host_path(path):
    return path if ROOT == "/" else os.path.join(ROOT, path.lstrip("/"))


def param_path(name):
    return host_path(os.path.join(PARAM_DIR, name))


def config_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "npu-status")


def config_file():
    return os.path.join(config_dir(), "queue.json")


def state_file():
    return os.path.join(config_dir(), "queue-state.json")


def stop_file():
    return STOP_FILE if ROOT == "/" else host_path(STOP_FILE)


def _read(name):
    try:
        with open(param_path(name), errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _bool(text):
    if text is None:
        return None
    return text[:1] in ("Y", "y", "1")


# --------------------------------------------------------------------------
# Reading


@dataclass
class Client:
    pid: int
    name: str
    priority: str
    applied: bool


@dataclass
class QueueState:
    loaded: bool = False        # etnaviv is loaded
    limit: int | None = None
    runtime: bool = False       # patch 0006: hw_job_limit can change while running
    priority: bool = False      # patch 0007: priority by name
    auto: bool | None = None
    turns: bool | None = None
    guard_ms: int | None = None
    trips: int = 0
    trip_from: int = 0
    table: dict = field(default_factory=dict)
    clients: list = field(default_factory=list)
    writable: bool = False      # this user may write hw_job_limit

    @property
    def tripped(self):
        return self.trips > 0


def parse_table(text):
    """'a=high,b=low' -> {'a': 'high', 'b': 'low'} (bad entries skipped, as the driver does)."""
    table = {}
    for item in (text or "").replace("\n", ",").split(","):
        name, sep, level = item.strip().rpartition("=")
        name, level = name.strip(), level.strip()
        if sep and name and level in LEVELS and name not in table:
            table[name] = level
    return table


def format_table(priorities):
    """The string for hw_job_prio. 'normal' is the default, so it is left out to save slots."""
    items = [(n, l) for n, l in sorted(priorities.items()) if l in LEVELS and l != "normal"]
    return ",".join(f"{n}={l}" for n, l in items[:MAX_NAMES])


def parse_clients(text):
    """hw_job_clients lines '<pid> <name> <priority> <applied|pending>'; names may hold spaces."""
    clients = []
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        pid = _int(parts[0])
        if pid is None:
            continue
        clients.append(Client(pid=pid, name=" ".join(parts[1:-2]), priority=parts[-2],
                              applied=parts[-1] == "applied"))
    return clients


def read_state(clients=True):
    st = QueueState()
    st.limit = _int(_read("hw_job_limit"))
    st.loaded = st.limit is not None
    if not st.loaded:
        return st
    st.auto = _bool(_read("hw_job_auto"))
    st.runtime = st.auto is not None
    st.turns = _bool(_read("hw_job_turns"))
    st.guard_ms = _int(_read("hw_job_prio_guard_ms"))
    st.trips = _int(_read("hw_job_trips")) or 0
    st.trip_from = _int(_read("hw_job_trip_from")) or 0
    table = _read("hw_job_prio")
    st.priority = table is not None
    st.table = parse_table(table)
    if clients and st.priority:
        st.clients = parse_clients(_read("hw_job_clients"))
    st.writable = st.runtime and os.access(param_path("hw_job_limit"), os.W_OK)
    return st


def summary(st):
    """One line for the indicator's menu, or None when there is nothing to say."""
    if not st.loaded:
        return None
    if not st.runtime:
        return f"Job queue: {st.limit} at once (fixed at boot)"
    parts = [f"up to {st.limit} at once"]
    if st.turns:
        parts.append("programs take turns")
    if st.trips:
        parts.append(f"⚠ dropped to 1 after a hang ({st.trips}×)")
    elif st.auto is False:
        parts.append("no automatic drop")
    return "Job queue: " + " · ".join(parts)


# --------------------------------------------------------------------------
# Writing


class NoAccess(Exception):
    """The user may not write the driver's parameters (root step 250 not run)."""


def write_param(name, value):
    if name not in WRITABLE:
        raise ValueError(f"{name} is not a writable job-queue parameter")
    try:
        with open(param_path(name), "w") as f:
            f.write(f"{value}\n")
    except PermissionError as e:
        raise NoAccess(f"no write access to {name} (root step 250 sets that up)") from e


def set_limit(n):
    n = int(n)
    if not LIMIT_MIN <= n <= LIMIT_MAX:
        raise ValueError(f"limit must be {LIMIT_MIN}-{LIMIT_MAX}")
    write_param("hw_job_limit", n)


def set_auto(on):
    write_param("hw_job_auto", "Y" if on else "N")


def set_turns(on):
    write_param("hw_job_turns", "Y" if on else "N")


def set_guard(ms):
    ms = int(ms)
    if not 0 <= ms <= GUARD_MAX_MS:
        raise ValueError(f"guard must be 0-{GUARD_MAX_MS} ms")
    write_param("hw_job_prio_guard_ms", ms)


def set_priorities(priorities):
    write_param("hw_job_prio", format_table(priorities))


def check_name(name):
    """None if the driver can match this program name, else why not."""
    if not name or not name.strip():
        return "the name is empty"
    if name != name.strip():
        return "the name starts or ends with a space"
    if len(name.encode()) > NAME_MAX:
        return f"the kernel keeps only {NAME_MAX} characters of a program's name"
    if any(ch in name for ch in ",=\n"):
        return "the name may not contain , = or a line break"
    return None


# --------------------------------------------------------------------------
# Saved settings


def _load_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(_load_json(config_file()))
    pri = cfg.get("priorities")
    cfg["priorities"] = {n: l for n, l in (pri.items() if isinstance(pri, dict) else [])
                         if isinstance(n, str) and l in LEVELS and check_name(n) is None}
    return cfg


def save_config(cfg):
    _save_json(config_file(), cfg)


def boot_id():
    try:
        with open(host_path(BOOT_ID)) as f:
            return f.read().strip()
    except OSError:
        return ""


def load_state():
    return _load_json(state_file())


def save_state(state):
    _save_json(state_file(), state)


def apply_saved(st=None, cfg=None):
    """Write the saved settings to the driver. Returns a list of messages for the user.

    The saved limit is not applied while the kit's STOP latch is set or after a trip in
    this boot: those keep the NPU at one job at a time until the user decides."""
    st = st or read_state(clients=False)
    cfg = cfg or load_config()
    msgs = []
    if not st.runtime:
        return msgs
    try:
        if st.priority and (cfg["priorities"] or st.table):
            set_priorities(cfg["priorities"])
        if cfg["auto"] is not None and cfg["auto"] != st.auto:
            set_auto(cfg["auto"])
        if cfg["turns"] is not None and cfg["turns"] != st.turns:
            set_turns(cfg["turns"])
        if cfg["guard_ms"] is not None and st.guard_ms is not None and cfg["guard_ms"] != st.guard_ms:
            set_guard(cfg["guard_ms"])
        want = cfg["limit"]
        if want is not None and want != st.limit:
            if os.path.exists(stop_file()):
                msgs.append(f"Kept the NPU queue at {st.limit}: NPU jobs are blocked after a hang "
                            f"(runs/STOP). Saved setting: {want}.")
            elif st.trips:
                msgs.append(f"Kept the NPU queue at {st.limit}: it dropped to 1 after a hang in this "
                            f"boot. Saved setting: {want}.")
            else:
                set_limit(want)
    except (NoAccess, ValueError, OSError) as e:
        msgs.append(f"Saved NPU queue settings not applied: {e}")
    return msgs


def apply_at_login():
    """apply_saved() once per boot, if the user asked for it. Returns messages (may be empty)."""
    cfg = load_config()
    if not cfg["apply_at_login"]:
        return []
    st = read_state(clients=False)
    if not st.runtime:
        return []
    state = load_state()
    boot = boot_id()
    if boot and state.get("applied_boot") == boot:
        return []
    msgs = apply_saved(st, cfg)
    state["applied_boot"] = boot
    save_state(state)
    return msgs


# --------------------------------------------------------------------------
# Trips


def unanswered_trip(st):
    """(trip_from, trips) if the driver tripped in this boot and the user hasn't answered."""
    if not st.trips:
        return None
    ack = load_state().get("trip_answered", {})
    if ack.get("boot") == boot_id() and int(ack.get("trips", 0)) >= st.trips:
        return None
    return st.trip_from, st.trips


def answer_trip(st):
    state = load_state()
    state["trip_answered"] = {"boot": boot_id(), "trips": st.trips}
    save_state(state)


class TripWatch:
    """Reports each new trip once, across indicator restarts (answers are saved)."""

    def __init__(self):
        self.reported = 0

    def check(self, st):
        trip = unanswered_trip(st)
        if trip is None or st.trips <= self.reported:
            return None
        self.reported = st.trips
        return trip


def reboot():
    """Ask systemd for a reboot (polkit decides whether a password is needed)."""
    subprocess.Popen(["systemctl", "reboot"], stdin=subprocess.DEVNULL, start_new_session=True)
