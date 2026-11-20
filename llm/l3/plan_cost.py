#!/usr/bin/env python3
"""plan_cost.py (2026-11-19, LLM engine, CPU only): per-token decode cost of CPU / NPU work splits, from measured numbers.

NPU (jb ladder 2026-11-19, runs/llm-jb-ladder-20261119-135828): fixed cost per submit F = 0.163 ms, weight streaming
inside a submit R_npu = 5.26 GB/s, each extra job in the same submit +35 us (4 groups) / +14 us (16 groups).
CPU: effective decode rate of llama.cpp Q8_0 on 4 A73s (cpu-baseline2: tokens/s x model bytes; jb_run's own int8 GEMV
measured 6.6-7.5 GB/s alone in the split ladders 2026-11-20, so these rates hold).
Side by side (MEASURED 2026-11-20, runs/llm-split-ladder2-20261120-011234, 1.7B gateup split): the NPU streams at 0.57
of its rate alone (5.46 -> 3.11 GB/s) and the CPU at 0.60 (6.68 -> 4.02); together 7.1-7.4 GB/s = the memory bus.
(The 2026-11-19 version used L1c's -8% / ~11 GB/s together, which these ladders did not reproduce.)
SPLIT_SYNC = 0.25 ms per split step, MEASURED: reading the NPU's int8 outputs back (Mesa reads its write-combined
buffer byte by byte: 0.23 ms for 1.5-6.6 KB on an A73, ~150 ns per byte on an A53) + the join.
A split step gives the NPU the first f of the output rows in one submit and the CPU the rest at the same time;
f is chosen so both finish together (f = 0 when the CPU alone is faster). Norms, RoPE, attention over the cache and SiLU are
not modelled (small at short contexts; llama.cpp's effective rate already contains them for the CPU-only line).
Usage: plan_cost.py [MODEL_DIR ...]
"""
import json
import os
import sys

F, R_NPU = 0.163, 5.26                  # ms, GB/s
EXTRA_JOB = {1: 0.0, 4: 0.035, 16: 0.014}
CPU_SLOW, NPU_SLOW, BW_CAP, SPLIT_SYNC = 0.60, 0.57, 7.4, 0.25   # measured 2026-11-20 (was 0.92, -, 11.0, 0.02)
R_CPU = {"SmolLM2-135M-Instruct": 6.4, "SmolLM2-360M-Instruct": 6.7, "SmolLM2-1.7B-Instruct": 7.6,
         "SmolLM3-3B": 6.0}             # GB/s effective (SmolLM3 from its Q4_K_M run: 5.0 tok/s prompt, 3.1 gen)
MODELS = "/home/jordan/Documents/AI_Projects/npu-llm/models"


def shapes(md):
    c = json.load(open(os.path.join(md, "config.json")))
    D, L, I = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"]
    H, KV = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or D // H
    steps = {"qkv": D * (H + 2 * KV) * hd, "o": H * hd * D, "gateup": 2 * D * I, "down": I * D}
    return L, steps, c["vocab_size"] * D


def npu_ms(b, groups=1):
    return F + b / 1e6 / R_NPU + (groups - 1) * EXTRA_JOB[groups]


def step_ms(b, where, rc, groups=1):
    """b bytes; where = cpu | npu | split. Returns (ms, NPU submits)."""
    cpu = b / 1e6 / rc
    if where == "cpu":
        return cpu, 0
    if where == "npu":
        return npu_ms(b, groups), 1
    rcs = rc * CPU_SLOW
    fixed = F + (groups - 1) * EXTRA_JOB[groups]
    tn, tc = b / 1e6 / (R_NPU * NPU_SLOW), b / 1e6 / rcs   # whole step on one side, streaming side by side
    f = (tc - fixed) / (tn + tc)
    if f <= 0:
        return cpu, 0
    t = max(fixed + f * tn, (1 - f) * tc)
    t = max(t, b / 1e6 / BW_CAP)                # shared-memory ceiling
    t += SPLIT_SYNC
    return (t, 1) if t < cpu else (cpu, 0)


PLANS = [  # name, placement per step (qkv, o, gateup, down, head), NPU groups
    ("CPU only (= llama.cpp rate)", dict(qkv="cpu", o="cpu", gateup="cpu", down="cpu", head="cpu"), 1),
    ("NPU all 4 steps, 1 group", dict(qkv="npu", o="npu", gateup="npu", down="npu", head="cpu"), 1),
    ("NPU all 4 steps, 16 groups", dict(qkv="npu", o="npu", gateup="npu", down="npu", head="cpu"), 16),
    ("NPU gateup only, rest CPU (serial)", dict(qkv="cpu", o="cpu", gateup="npu", down="cpu", head="cpu"), 4),
    ("split gateup (H2), 4 groups", dict(qkv="cpu", o="cpu", gateup="split", down="cpu", head="cpu"), 4),
    ("split gateup (H2), 16 groups", dict(qkv="cpu", o="cpu", gateup="split", down="cpu", head="cpu"), 16),
    ("split gateup + down (H1), 4 groups", dict(qkv="cpu", o="cpu", gateup="split", down="split", head="cpu"), 4),
    ("split every step + head, 4 groups", dict(qkv="split", o="split", gateup="split", down="split", head="split"), 4),
]


def main():
    dirs = sys.argv[1:] or [os.path.join(MODELS, m) for m in R_CPU]
    for md in dirs:
        name = os.path.basename(md.rstrip("/"))
        L, st, head = shapes(md)
        rc = R_CPU[name]
        total = L * sum(st.values()) + head
        print("\n%s: %d layers, %.0f MB int8 (layers %.0f + head %.0f), CPU %.1f GB/s effective" % (
            name, L, total / 1e6, L * sum(st.values()) / 1e6, head / 1e6, rc))
        print("  per layer: " + ", ".join("%s %.2f MB" % (k, v / 1e6) for k, v in st.items()))
        base = None
        for pname, place, g in PLANS:
            ms, subs = 0.0, 0
            for k, b in st.items():
                t, s = step_ms(b, place[k], rc, g if k in ("gateup", "down") or place[k] != "cpu" else 1)
                ms += L * t
                subs += L * s
            t, s = step_ms(head, place["head"], rc, g)
            ms += t
            subs += s
            base = base or ms
            print("  %-38s %6.1f ms/token  %5.1f tok/s  %4d submits  x%.2f" % (pname, ms, 1000 / ms, subs, base / ms))


if __name__ == "__main__":
    main()
