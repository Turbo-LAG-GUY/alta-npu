"""engine.py (2026-11-24): Alta Chat's side of the chat engine (jb_prefill --serve, llm/l3/jb_prefill.c).

The engine runs as its own systemd user unit (alta-chat-engine), so the model stays loaded if the window closes
or crashes, and it never inherits the window's environment (the window runs under the NPU guard). NPU mode goes
through scripts/npu-attempt -> npu-run, like every NPU job on this board: the global NPU lock, the kernel-error
watch, the STOP latch and a PROGRESS.md record. The engine ends itself before npu-run's deadline (--max-s), so
a long chat never trips the deadline latch.
"""
import json
import os
import socket
import subprocess
import sys

sys.path.append("/home/jordan/Documents/AI_Projects/.venv/lib/python3.12/site-packages")   # tokenizers only
from tokenizers import Tokenizer  # noqa: E402

AI = "/home/jordan/Documents/AI_Projects"
KIT = "/home/jordan/alta-npu"
ENGINE = AI + "/npu-llm/bin/jb_prefill-21e4e4e9"
UNIT = "alta-chat-engine"
RUNDIR = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "alta-chat")
SOCK = os.path.join(RUNDIR, "engine.sock")
STATUS = SOCK + ".status"

MODELS = {
    "SmolLM2 135M": dict(packs=AI + "/npu-llm/models/jb/real-135m", aux=AI + "/npu-llm/models/jb/aux-135m/model.jba",
                         tok=AI + "/npu-llm/models/SmolLM2-135M-Instruct/tokenizer.json", ctx=2048, label="135M"),
    "SmolLM2 1.7B": dict(packs=AI + "/npu-llm/models/jb/real-1p7b-split4",
                         aux=AI + "/npu-llm/models/jb/aux-1p7b/model.jba",
                         tok=AI + "/npu-llm/models/SmolLM2-1.7B-Instruct/tokenizer.json", ctx=1024, label="1p7b"),
}
# the driver settings every passing LLM ladder used (prefill5-ladder.sh)
NPU_ENV = ["ETNA_ML_KEEP_IMAGE_CACHE=1", "ETNA_ML_V7_TILE_FIT=262144", "ETNA_ML_EXACT_BO=1",
           "ETNA_ML_CACHED_COEFFS=1"]
SESSION_S = 3 * 3600          # the engine stops itself after this long; npu-run's deadline is 30 min later
IM_START, IM_END = 1, 2
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def preflight(backend):
    """reasons not to start (empty list: OK)"""
    bad = []
    if not os.access(ENGINE, os.X_OK):
        bad.append("engine binary missing: " + ENGINE)
    if backend == "npu":
        if os.path.exists(KIT + "/runs/STOP"):
            bad.append("runs/STOP is set (an earlier NPU job hit a kernel error): Jordan reviews and clears it")
        if subprocess.run(["pgrep", "-i", "steam"], capture_output=True).returncode == 0:
            bad.append("Steam is running: close it first (it can land on the NPU)")
    return bad


def unit_active():
    r = subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating")


def start(model, backend):
    m = MODELS[model]
    os.makedirs(RUNDIR, exist_ok=True)
    for p in (SOCK, STATUS):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    eng = [ENGINE, "--npu" if backend == "npu" else "--ref", "--serve", SOCK, "--ctx", str(m["ctx"]),
           "--max-s", str(SESSION_S), m["packs"], m["aux"]]
    if backend == "npu":
        cmd = ["env"] + NPU_ENV + ["OMP_NUM_THREADS=4", "NPU_DEADLINE=%d" % (SESSION_S + 1800),
                                   KIT + "/scripts/npu-attempt", "CHAT-" + m["label"], "--"] + eng
    else:
        cmd = ["env", "OMP_NUM_THREADS=4"] + eng
    subprocess.run(["systemctl", "--user", "reset-failed", UNIT], capture_output=True)
    return subprocess.run(["systemd-run", "--user", "--unit=" + UNIT, "--collect", "--working-directory=" + KIT,
                           "--description=Alta Chat engine (%s, %s)" % (model, backend)] + cmd,
                          capture_output=True, text=True)


def status():
    try:
        return open(STATUS).read().strip()
    except OSError:
        return ""


def journal_tail(n=15):
    r = subprocess.run(["journalctl", "--user", "-u", UNIT, "-n", str(n), "--no-pager", "-o", "cat"],
                       capture_output=True, text=True)
    return r.stdout.strip()


class Client:
    """one connection to the engine; lines in, lines out"""

    def __init__(self):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(SOCK)
        self.f = self.s.makefile("rb")
        ready = self.f.readline().decode().strip()
        if not ready.startswith("READY "):
            raise RuntimeError("engine said: " + ready)
        self.info = json.loads(ready[6:])

    def send(self, line):
        self.s.sendall((line + "\n").encode())

    def gen(self, ids, max_new=256, temp=0.7, top_k=40, top_p=0.9, rep_pen=1.1, seed=0):
        """yields ("PREFILL", ms, path, new) / ("TOK", id) / ("END", reason, n, ms_per_token, prefill_ms)"""
        self.send("GEN %d %g %d %g %g %d %d %s" % (max_new, temp, top_k, top_p, rep_pen, seed, len(ids),
                                                    " ".join(map(str, ids))))
        while True:
            ln = self.f.readline().decode()
            if not ln:
                raise RuntimeError("the engine closed the connection")
            w = ln.split()
            if w[0] == "TOK":
                yield ("TOK", int(w[1]))
            elif w[0] == "PREFILL":
                yield ("PREFILL", float(w[1]), w[2], int(w[3]))
            elif w[0] == "END":
                yield ("END", w[1], int(w[2]), float(w[3]), float(w[4]))
                return
            elif w[0] == "ERR":
                raise RuntimeError(ln[4:].strip())

    def stop(self):
        self.send("STOP")

    def quit(self):
        try:
            self.send("QUIT")
        except OSError:
            pass
        self.s.close()


class Conversation:
    """token ids of a chat, built piece by piece (the assistant's own ids kept verbatim, so the engine's cached
    prefix always matches)"""

    def __init__(self, tok_path, system=SYSTEM):
        self.tk = Tokenizer.from_file(tok_path)
        self.system = system
        self.reset()

    def enc(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def reset(self):
        self.ids = [IM_START] + self.enc("system\n" + self.system) + [IM_END] + self.enc("\n")

    def add_user(self, text):
        self.ids += [IM_START] + self.enc("user\n" + text) + [IM_END] + self.enc("\n")
        self.ids += [IM_START] + self.enc("assistant\n")

    def add_reply(self, toks):
        self.ids += list(toks) + ([] if toks and toks[-1] == IM_END else [IM_END]) + self.enc("\n")

    def text(self, toks):
        return self.tk.decode([t for t in toks if t not in (0, IM_END)], skip_special_tokens=False)
