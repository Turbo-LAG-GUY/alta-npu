#!/usr/bin/env python3
"""export_aux.py (2026-11-23): everything jb_prefill (the prompt-chunk loop in C) needs besides the NPU packs, and a
reference run to check it against.
  OUT/model.jba   the rotated model's small parts: config, bf16 embeddings (= the tied head), the residual rotation
                  (randomized-Hadamard signs, or a dense matrix), the down-input block Hadamard signs, the final norm
                  weight, RoPE inverse frequencies, which layers use RoPE, and each step's job row maps (pack output
                  order -> model rows)
  OUT/<text>.jbp0 one prompt: its 257 token ids and position 0's K and V per layer (position 0 only attends to
                  itself, so tokens 1.. need nothing else from it; in a chat it is always the same template token)
  OUT/<text>.ref  the reference: the C-glue Python path (chunk_glue --glue casync, int8 reference backend) - logits
                  of every position and the residual after every layer, to compare jb_prefill --ref with
File format JBA1: "JBA1", then records: name (48 bytes, NUL-padded), u32 dtype (0 f32, 1 f64, 2 i32, 3 u16, 4 i8),
u32 ndim, u32 dims[4], u64 nbytes, data.
Usage: export_aux.py MODEL_DIR PACK_DIR OUT_DIR [text ...]   (texts: gpl3 readme; default gpl3)"""
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = {np.dtype(np.float32): 0, np.dtype(np.float64): 1, np.dtype(np.int32): 2, np.dtype(np.uint16): 3,
      np.dtype(np.int8): 4}


def write_jba(path, recs):
    with open(path, "wb") as f:
        f.write(b"JBA1")
        for name, a in recs:
            a = np.ascontiguousarray(a)
            dims = list(a.shape) + [1] * (4 - a.ndim)
            f.write(name.encode().ljust(48, b"\0") + struct.pack("<II4IQ", DT[a.dtype], a.ndim, *dims, a.nbytes))
            f.write(a.tobytes())
    print("wrote %s (%d records, %.1f MB)" % (path, len(recs), os.path.getsize(path) / 1e6), flush=True)


def main():
    MD, PD, OUT = (os.path.abspath(p) for p in sys.argv[1:4])
    texts = sys.argv[4:] or ["gpl3"]
    os.makedirs(OUT, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("GOMP_CPU_AFFINITY", "2-5")
    sys.argv = [sys.argv[0], MD]
    sys.path.insert(0, os.path.join(HERE, ".."))
    sys.path.insert(0, os.path.join(HERE, "..", "..", "ref"))
    sys.path.insert(0, HERE)
    import json
    import l2g_outliers as G
    import l2_quality
    from smollm import SmolLM
    from tokenizers import Tokenizer
    import chunk_glue as CG
    import cforward as CF
    t0 = time.monotonic()
    man = json.load(open(os.path.join(PD, "manifest.json")))
    gids = np.load(os.path.join(PD, "groups.npz"))
    m = SmolLM(MD)
    what = G.rotate(m, m.layers)
    assert what == man["rotation"], (what, man["rotation"])
    Q, Hd = m.post_embed.__self__, m.down_in.__self__
    TOK = 256
    steps, order = {}, {}
    for li in range(m.L):
        for s in CG.STEPS:
            key = "%d.%s" % (li, s)
            steps[key] = CG.Step(os.path.join(PD, "L%02d_%s_t%d.jbp" % (li, s, TOK)), gids, key, TOK)
            order[key] = len(order)
    I = m.cfg["intermediate_size"]
    V = m.embed.shape[0]
    cfg = np.array([m.L, m.D, m.H, m.KV, m.hd, I, V, 1 if Q.pow2 else 0, Hd.b, TOK], np.int32)
    recs = [("cfg", cfg), ("eps", np.array([m.eps], np.float32)),
            ("embed", np.asarray(m.embed.mm, np.uint16).reshape(V, m.D) if m.embed.dtype == "BF16" else
             (m.embed.f32().view(np.uint32) >> 16).astype(np.uint16)),
            ("Q", Q.s if Q.pow2 else Q.Q), ("Hs", Hd.s), ("norm", m.norm.astype(np.float32)),
            ("inv_freq", np.asarray(m.inv_freq, np.float64)), ("rope", np.asarray(m.use_rope, np.int32))]
    for key, st in steps.items():
        li, s = key.split(".")
        for j, i in enumerate(st.idx):
            recs.append(("rows.%s.%s.%d" % (li, s, j), np.ascontiguousarray(i, np.int32)))
    write_jba(os.path.join(OUT, "model.jba"), recs)
    tok = Tokenizer.from_file(MD + "/tokenizer.json")
    l2_quality.T = TOK + 1
    _, evals = l2_quality.texts(tok)
    for tn in texts:
        ids = evals[tn]
        q = CF.PackQuantC(steps, order, CG.RefBackend())
        rec = {"k0": [], "v0": [], "x": []}
        lg = CF.forward_c_async(m, ids, q, record=rec)
        write_jba(os.path.join(OUT, "%s.jbp0" % tn), [("ids", np.asarray(ids, np.int32))] +
                  [("k0.%d" % li, k) for li, k in enumerate(rec["k0"])] +
                  [("v0.%d" % li, v) for li, v in enumerate(rec["v0"])])
        write_jba(os.path.join(OUT, "%s.ref" % tn), [("logits", lg.astype(np.float32))] +
                  [("x.%d" % li, x) for li, x in enumerate(rec["x"])])
    print("export_aux: %.0f s" % (time.monotonic() - t0))


if __name__ == "__main__":
    main()
