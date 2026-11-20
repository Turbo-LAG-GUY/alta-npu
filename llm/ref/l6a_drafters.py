#!/usr/bin/env python3
"""l6a_drafters.py (2026-11-19, CPU only): replay draft predictors over SmolLM2's saved greedy answers (L6a).

Greedy speculative decoding: each step the drafter proposes up to K tokens, the target checks all of them in one
pass over the weights (M = K + 1 positions), keeps the longest matching prefix and adds its own next token. So a
step advances accepted + 1 tokens, and the final text is exactly the greedy answer (replay is exact).

Drafters
  lookup     prompt lookup: the tokens that followed the latest earlier occurrence of the last n (4..1) tokens
  tage       TAGE-style: tables over the last 1, 2, 3, 4, 6, 8 tokens -> (next token, 2-bit confidence), learned
             online from the conversation (prompt + answer so far); provider = longest matching history, with
             TAGE's alternate prediction when the provider entry is new (confidence 0)
  tage+warm  same tables first trained on the "train" answers and kept across conversations (like a branch
             predictor that stays warm)
  +perc      a perceptron (Jimenez-style training) decides at each draft position whether to keep drafting,
             from: provider history length, its confidence, alt used, draft position, token seen in the prompt
Cost of a check with M positions, relative to one normal step (L1b: M = 16 as a 4x4 image costs r = 2.1-4.3x):
  linear  c(M) = 1 + (r - 1)(M - 1)/15        fixed  c(M) = r for any M > 1 (only the 16-token job exists)
Speedup = answer tokens / sum of step costs (normal decoding = 1 per token).
"""
import json
import sys
from collections import defaultdict

import numpy as np

LENS = (1, 2, 3, 4, 6, 8)


class Tage:
    def __init__(self):
        self.t = {n: {} for n in LENS}

    def learn(self, hist, tok):
        for n in LENS:
            if len(hist) < n:
                continue
            k = tuple(hist[-n:])
            e = self.t[n].get(k)
            if e is None:
                self.t[n][k] = [tok, 0]
            elif e[0] == tok:
                e[1] = min(e[1] + 1, 3)
            elif e[1] > 0:
                e[1] -= 1
            else:
                e[0] = tok

    def learn_seq(self, seq, start=0):
        for i in range(max(start, 1), len(seq)):
            self.learn(seq[max(0, i - 8):i], seq[i])

    def predict(self, hist):
        """-> (token, provider length, confidence, alt used) or None."""
        hits = []
        for n in reversed(LENS):
            if len(hist) >= n:
                e = self.t[n].get(tuple(hist[-n:]))
                if e is not None:
                    hits.append((n, e))
        if not hits:
            return None
        n, e = hits[0]
        if e[1] == 0:
            for n2, e2 in hits[1:]:
                if e2[1] > 0:
                    return e2[0], n2, e2[1], True
        return e[0], n, e[1], False


def lookup_draft(ctx, K):
    for n in (4, 3, 2, 1):
        if len(ctx) <= n:
            continue
        key = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):
            if ctx[i:i + n] == key:
                return ctx[i + n:i + n + K]
    return []


def features(p, j, in_prompt):
    x = np.zeros(7 + 4 + 2 + 6 + 2 + 1)
    n, conf, alt = p[1], p[2], p[3]
    x[LENS.index(n)] = 1
    x[7 + conf] = 1
    x[11 + int(alt)] = 1
    x[13 + min(j, 5)] = 1
    x[19 + int(in_prompt)] = 1
    x[21] = 1
    return x


class Perceptron:
    def __init__(self, dim=22, theta=2.0):
        self.w, self.theta, self.bias = np.zeros(dim), theta, 0.0

    def out(self, x):
        return float(self.w @ x)

    def train(self, x, accepted):
        y, t = self.out(x), (1 if accepted else -1)
        if (y >= 0) != accepted or abs(y) <= self.theta:
            self.w += 0.1 * t * x


def tage_draft(tg, ctx, K, prompt_set, perc=None, record=None):
    hist, out = list(ctx[-8:]), []
    for j in range(K):
        p = tg.predict(hist)
        if p is None:
            break
        if perc is not None and perc.out(features(p, j, p[0] in prompt_set)) < perc.bias:
            break
        if record is not None:
            record.append(features(p, j, p[0] in prompt_set))
        out.append(p[0])
        hist = (hist + [p[0]])[-8:]
    return out


def replay(ans, kind, K, tg=None, perc=None, train_perc=None):
    """-> list of (accepted, drafted) per step."""
    ctx, gen = list(ans["prompt_ids"]), ans["out_ids"]
    pset = set(ctx)
    if kind.startswith("tage"):
        tg.learn_seq(ctx)
    steps, t = [], 0
    while t < len(gen):
        cur = ctx + gen[:t]
        rec = [] if train_perc is not None else None
        if kind == "lookup":
            d = lookup_draft(cur, K)
        elif kind.startswith("tage"):
            d = tage_draft(tg, cur, K, pset, perc, rec)
        else:
            d = []
        a = 0
        while a < len(d) and t + a < len(gen) and d[a] == gen[t + a]:
            a += 1
        if train_perc is not None:
            for j, x in enumerate(rec):
                if j > a:
                    break
                train_perc.train(x, j < a)
        adv = min(a + 1, len(gen) - t)
        if kind.startswith("tage"):
            full = ctx + gen
            for i in range(len(ctx) + t, len(ctx) + t + adv):
                tg.learn(full[max(0, i - 8):i], full[i])
        steps.append((a, len(d)))
        t += adv
    return steps


def cost(steps, r, model):
    c = 0.0
    for a, d in steps:
        m = d + 1
        if m == 1:
            c += 1
        elif model == "linear":
            c += 1 + (r - 1) * (m - 1) / 15
        else:
            c += r
    return c


def main():
    data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                          "/home/jordan/Documents/AI_Projects/npu-llm/runs/l6a-greedy.json"))
    train = [a for a in data if a["split"] == "train"]
    test = [a for a in data if a["split"] == "test"]
    ntok = sum(len(a["out_ids"]) for a in test)
    print("test: %d answers, %d tokens; train: %d answers, %d tokens" % (
        len(test), ntok, len(train), sum(len(a["out_ids"]) for a in train)))
    cats = sorted(set(a["cat"] for a in test))
    # perceptron trained on the train answers with a warm TAGE (two passes)
    perc = Perceptron()
    for _ in range(2):
        tg = Tage()
        for a in train:
            replay(a, "tage", 15, tg, None, perc)
    warm = Tage()
    for a in train:
        warm.learn_seq(a["prompt_ids"] + a["out_ids"])
    rows = []
    for name, kind, K, usewarm, useperc in (("lookup K=15", "lookup", 15, False, False),
                                             ("lookup K=7", "lookup", 7, False, False),
                                             ("lookup K=3", "lookup", 3, False, False),
                                             ("tage K=15", "tage", 15, False, False),
                                             ("tage+warm K=15", "tage", 15, True, False),
                                             ("tage+warm K=7", "tage", 7, True, False),
                                             ("tage+warm+perc K=15", "tage", 15, True, True)):
        import copy
        tg = copy.deepcopy(warm) if usewarm else None
        per = defaultdict(list)
        for a in test:
            if not usewarm and kind == "tage":
                tg = Tage()
            per[a["cat"]].append(replay(a, kind, K, tg, perc if useperc else None))
        rows.append((name, per))
    hdr = "%-22s %-8s %7s %8s %9s" % ("drafter", "category", "tok/step", "acc/try", "drafted")
    print("\n" + hdr + "   speedup linear r=2 / r=3 / r=4   fixed r=3")
    for name, per in rows:
        for cat in cats + ["ALL"]:
            steps = [s for c in (cats if cat == "ALL" else [cat]) for ans in per[c] for s in ans]
            toks = sum(len(x["out_ids"]) for x in test if cat in ("ALL", x["cat"]))
            tries = [a for a, d in steps if d > 0]
            sp = [toks / cost(steps, r, "linear") for r in (2, 3, 4)]
            print("%-22s %-8s %7.2f %8.2f %8.1f%%   %6.2fx %6.2fx %6.2fx   %6.2fx" % (
                name if cat == cats[0] else "", cat, toks / len(steps), np.mean(tries) if tries else 0,
                100 * len(tries) / len(steps), sp[0], sp[1], sp[2], toks / cost(steps, 3, "fixed")))
        print()


if __name__ == "__main__":
    sys.exit(main())
