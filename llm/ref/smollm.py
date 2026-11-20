#!/usr/bin/env python3
"""smollm.py (2026-11-18, CPU only): plain numpy SmolLM2 (Llama architecture) for the LLM engine's step L2.

Three jobs:
  1. fp32 reference (the quality bar; later the engine's per-layer check values),
  2. NPU 8-bit simulation: every linear layer as the NN unit would do it - per-tensor int8 weights (symmetric),
     per-tensor int8 input and output with zero points (the job config holds ONE scale each), exact integer
     accumulation; optionally q/k/v and gate/up fused into one job each (shared weight and output scale),
  3. FFN statistics: how many neurons per token carry the output (sparsity), and how much the active set repeats
     across neighbouring tokens (what a TAGE-style history predictor could use).
Weights: model.safetensors (bf16) read with the stdlib + numpy, no torch.
"""
import json
import os
import struct

import numpy as np

LINEARS = ("q", "k", "v", "o", "gate", "up", "down")
HF = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
      "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
FUSED = {"qkv": ("q", "k", "v"), "gateup": ("gate", "up")}


class Lazy:
    """A weight tensor left in the (memory-mapped) safetensors file; .f32() expands it when it is used, so RAM
    holds one layer at a time (the page cache keeps small models; big ones page from the card)."""

    def __init__(self, mm, dtype, shape):
        self.mm, self.dtype, self.shape = mm, dtype, tuple(shape)

    def f32(self):
        if self.dtype == "BF16":
            return (self.mm.astype(np.uint32) << 16).view(np.float32).reshape(self.shape)
        return np.asarray(self.mm, np.float32).reshape(self.shape)

    def rows(self, idx):
        r = self.mm.reshape(self.shape)[idx]
        return (r.astype(np.uint32) << 16).view(np.float32) if self.dtype == "BF16" else r.astype(np.float32)


def load_safetensors(path):
    """name -> Lazy (one file or a sharded model.safetensors.index.json)."""
    paths = [path]
    idx = os.path.join(os.path.dirname(path), "model.safetensors.index.json")
    if not os.path.exists(path) and os.path.exists(idx):
        paths = sorted(set(os.path.join(os.path.dirname(path), v)
                           for v in json.load(open(idx))["weight_map"].values()))
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        for name, d in header.items():
            if name == "__metadata__":
                continue
            a, b = d["data_offsets"]
            dt = {"BF16": np.uint16, "F32": np.float32}[d["dtype"]]
            mm = np.memmap(p, dtype=dt, mode="r", offset=base + a, shape=((b - a) // np.dtype(dt).itemsize,))
            out[name] = Lazy(mm, d["dtype"], d["shape"])
    return out


def _w(W):
    """A weight given as an array, a Lazy tensor or a function returning an array -> fp32 array."""
    if isinstance(W, Lazy):
        return W.f32()
    return W() if callable(W) else W


class Quant:
    """How each linear layer is computed. mode: fp32 | w8 (int8 weights only) | w8a8 (the NPU: int8 in/out too).
    act: dict name -> (lo, hi) of the input/output ranges from calibrate() (static scales, as a compiled job);
    None = dynamic per call (our own engine could rewrite a job's scales per token)."""

    WQ = {}  # shared by every instance: key -> (int8 weights, scale); expanded to float64 only per call
    CACHE = True   # False for big models: quantize each weight when it is used, keep nothing
    F64 = True     # exact int32 accumulation in float64; False = float32 (relative error ~1e-7, half the memory)
    WGROUPS = 1    # weight scales per layer: output channels sorted by max |w|, split into this many groups
                   # (each group = its own NPU job with its own scale); 0 = one scale per channel
    WGROUPS_BY = None  # optional dict: step name (qkv o gateup down head ...) -> groups, overrides WGROUPS

    def __init__(self, mode="fp32", fused=False, act=None, quant_head=True, a8_layers=None, a8_side="both",
                 fp_first=False):
        """a8_layers: names (q k v o gate up down qkv gateup head) that get 8-bit activations in w8a8 mode, the
        rest stay w8 (None = all); a8_side: in | out | both = which side of those layers is quantized.
        fp_first: sequence position 0 (the attention sink with the massive activations) is computed like w8
        (the CPU's job), every later position in the 8-bit format; use with calibrate(..., skip_first=True)."""
        self.mode, self.fused, self.act, self.quant_head = mode, fused, act, quant_head
        self.a8_layers, self.a8_side, self.fp_first = a8_layers, a8_side, fp_first
        self.start = 1

    @staticmethod
    def qparams(lo, hi):
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        s = max(hi - lo, 1e-8) / 255.0
        return s, int(np.clip(np.round(-lo / s) - 128, -128, 127))

    @staticmethod
    def q8(x, s, zp):
        return np.clip(np.round(x / s) + zp, -128, 127)

    def weight(self, key, W):
        g = Quant.WGROUPS
        if Quant.WGROUPS_BY is not None:
            g = Quant.WGROUPS_BY.get(key.split(".")[-1], g)
        key = key if g == 1 else "%s#g%d" % (key, g)
        if key not in Quant.WQ:
            W = _w(W)
            if g == 1:
                s = np.float32(np.abs(W).max() / 127.0)
            else:
                r = np.abs(W).max(1)
                s = np.empty(W.shape[0], np.float32)
                for grp in np.array_split(np.argsort(r), g if g > 0 else W.shape[0]):
                    s[grp] = r[grp].max() / 127.0
            sc = np.asarray(s, np.float32).reshape(-1, 1)
            wq = (np.clip(np.round(W / sc), -127, 127).astype(np.int8), np.asarray(s, np.float32))
            if not Quant.CACHE:
                return wq
            Quant.WQ[key] = wq
        return Quant.WQ[key]

    def rng(self, key, x):
        if self.act is not None and key in self.act:
            return self.act[key]
        return float(x.min()), float(x.max())

    def linear(self, key, x, W):
        """key: 'L.name' (or 'L.qkv' / 'L.gateup' when fused, W stacked); x [T, IC]; W [OC, IC] fp32."""
        if self.mode == "fp32" or (key == "head" and not self.quant_head):
            return x @ _w(W).T
        Wq, sw = self.weight(key, W)
        lname = key.split(".")[-1]
        if self.mode == "w8" or (self.a8_layers is not None and lname not in self.a8_layers):
            return (x @ (Wq.astype(np.float32) * sw.reshape(-1, 1)).T).astype(np.float32)
        if self.fp_first and self.start == 0:
            y0 = (x[:1] @ (Wq.astype(np.float32) * sw.reshape(-1, 1)).T).astype(np.float32)
            if x.shape[0] == 1:
                return y0
            self.start = 1
            try:
                return np.concatenate([y0, self.linear(key, x[1:], W)])
            finally:
                self.start = 0
        if self.a8_side in ("in", "both"):
            sx, zx = self.qparams(*self.rng(key + ".in", x))
            xq = self.q8(x, sx, zx) - zx                 # exact small integers
            ft = np.float64 if Quant.F64 else np.float32
            acc = xq.astype(ft) @ Wq.astype(ft).T   # int32 accumulation (exact in float64)
            y = acc * (sx * sw.reshape(1, -1))
        else:
            y = (x.astype(np.float64) @ Wq.astype(np.float64).T) * sw.reshape(1, -1)
        if self.a8_side in ("out", "both"):
            sy, zy = self.qparams(*self.rng(key + ".out", y))
            return ((self.q8(y, sy, zy) - zy) * sy).astype(np.float32)
        return y.astype(np.float32)


class SmolLM:
    def __init__(self, model_dir):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        c = self.cfg
        self.L, self.D = c["num_hidden_layers"], c["hidden_size"]
        self.H, self.KV = c["num_attention_heads"], c["num_key_value_heads"]
        self.hd = self.D // self.H
        self.eps = c["rms_norm_eps"]
        self.hd = c.get("head_dim") or self.hd
        w = load_safetensors(os.path.join(model_dir, "model.safetensors"))
        self.embed = w["model.embed_tokens.weight"]           # Lazy; rows looked up, whole matrix = tied head
        self.norm = w["model.norm.weight"].f32()
        self._head = None
        self.layers = []
        for i in range(self.L):
            p = "model.layers.%d." % i
            d = {n: w[p + HF[n] + ".weight"] for n in LINEARS}
            d["ln1"], d["ln2"] = w[p + "input_layernorm.weight"].f32(), w[p + "post_attention_layernorm.weight"].f32()
            d["qkv"] = (lambda d=d: np.concatenate([d["q"].f32(), d["k"].f32(), d["v"].f32()]))
            d["gateup"] = (lambda d=d: np.concatenate([d["gate"].f32(), d["up"].f32()]))
            self.layers.append(d)
        # SmolLM3: every 4th layer has no RoPE (config no_rope_layers: 1 = RoPE, 0 = none)
        self.use_rope = c.get("no_rope_layers") or [1] * self.L
        inv = c["rope_theta"] ** (-np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        self.inv_freq = inv
        # optional hooks (None = off), used by l2g_outliers.py's rotations: x after the embedding lookup, the residual
        # before the final norm, the FFN activations right before down_proj (after ffn_hook)
        self.post_embed = self.pre_head = self.down_in = None

    def head(self):
        """Tied output head = the embedding matrix, fp32; kept when it is small (<= 256 MB)."""
        if self._head is not None:
            return self._head
        h = self.embed.f32()
        if h.nbytes <= 256 << 20:
            self._head = h
        return h

    def rms(self, x, w):
        return (x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + self.eps)).astype(np.float32) * w

    def rope(self, x, pos):  # x [T, heads, hd], non-interleaved (rotate_half)
        f = np.outer(pos, self.inv_freq)
        cos, sin = np.cos(np.concatenate([f, f], -1))[:, None], np.sin(np.concatenate([f, f], -1))[:, None]
        h = self.hd // 2
        rot = np.concatenate([-x[..., h:], x[..., :h]], -1)
        return (x * cos + rot * sin).astype(np.float32)

    def forward(self, ids, quant=None, cache=None, ffn_hook=None):
        """ids [T] -> logits [T, V]. cache: list of (K, V) per layer, extended in place (pass [] to start one).
        ffn_hook(layer, h) -> h' sees/changes the FFN activations silu(gate)*up [T, 1536] before down_proj."""
        q8 = quant or Quant()
        T = len(ids)
        start = cache[0][0].shape[0] if cache else 0
        q8.start = start
        pos = np.arange(start, start + T)
        x = self.embed.rows(np.asarray(ids))
        if self.post_embed is not None:
            x = self.post_embed(x)
        for li, d in enumerate(self.layers):
            a = self.rms(x, d["ln1"])
            if q8.fused:
                qkv = q8.linear("%d.qkv" % li, a, d["qkv"])
                q, k, v = np.split(qkv, [self.H * self.hd, (self.H + self.KV) * self.hd], axis=1)
            else:
                q, k, v = (q8.linear("%d.%s" % (li, n), a, d[n]) for n in "qkv")
            q = q.reshape(T, self.H, self.hd)
            k = k.reshape(T, self.KV, self.hd)
            if self.use_rope[li]:
                q, k = self.rope(q, pos), self.rope(k, pos)
            v = v.reshape(T, self.KV, self.hd)
            if cache is not None:
                if len(cache) <= li:
                    cache.append((k, v))
                else:
                    cache[li] = (np.concatenate([cache[li][0], k]), np.concatenate([cache[li][1], v]))
                k, v = cache[li]
            S = k.shape[0]
            g = self.H // self.KV
            kk, vv = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)       # [S, H, hd]
            sc = np.einsum("thd,shd->hts", q, kk) / np.sqrt(self.hd)
            mask = (np.arange(S)[None, :] > (pos[:, None]))                    # causal
            sc = np.where(mask[None], -1e30, sc)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            att = np.einsum("hts,shd->thd", sc, vv).reshape(T, self.H * self.hd).astype(np.float32)
            x = x + q8.linear("%d.o" % li, att, d["o"])
            a = self.rms(x, d["ln2"])
            if q8.fused:
                gu = q8.linear("%d.gateup" % li, a, d["gateup"])
                gt, up = np.split(gu, 2, axis=1)
            else:
                gt, up = q8.linear("%d.gate" % li, a, d["gate"]), q8.linear("%d.up" % li, a, d["up"])
            h = (gt / (1.0 + np.exp(-gt)) * up).astype(np.float32)
            if ffn_hook is not None:
                h = ffn_hook(li, h)
            if self.down_in is not None:
                h = self.down_in(h)
            x = x + q8.linear("%d.down" % li, h, d["down"])
        if self.pre_head is not None:
            x = self.pre_head(x)
        x = self.rms(x, self.norm)
        if (q8.mode == "fp32" or not q8.quant_head) and self.embed.shape[0] * self.embed.shape[1] * 4 > 256 << 20:
            # big vocabulary, full-precision head: in slices of 8192 rows, no 1 GB temporary
            V = self.embed.shape[0]
            return np.concatenate([x @ self.embed.rows(np.arange(a, min(a + 8192, V))).T for a in range(0, V, 8192)], 1)
        return q8.linear("head", x, self.head)


def calibrate(model, id_lists, fused, skip_first=False):
    """Record every linear's input and output range over the calibration texts (fp32 run).
    skip_first: leave sequence position 0 out (it runs in full precision with Quant(fp_first=True))."""
    rec = {}

    class Rec(Quant):
        def linear(self, key, x, W):
            y = x @ _w(W).T
            pairs = ((key + ".in", x), (key + ".out", y))
            if skip_first and self.start == 0:
                if x.shape[0] == 1:
                    return y
                pairs = ((key + ".in", x[1:]), (key + ".out", y[1:]))
            for k2, t in pairs:
                lo, hi = rec.get(k2, (np.inf, -np.inf))
                rec[k2] = (min(lo, float(t.min())), max(hi, float(t.max())))
            return y
    r = Rec(fused=fused)
    for ids in id_lists:
        model.forward(ids, r)
    return rec


def chat_ids(tok, user, system=None):
    s = "<|im_start|>system\n%s<|im_end|>\n" % (system or "You are a helpful AI assistant named SmolLM, trained by Hugging Face")
    s += "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % user
    return tok.encode(s, add_special_tokens=False).ids


def generate(model, ids, n, quant=None, stop=2):
    cache = []
    logits = model.forward(ids, quant, cache)
    out = []
    for _ in range(n):
        t = int(np.argmax(logits[-1]))
        if t == stop or t == 2:
            break
        out.append(t)
        logits = model.forward([t], quant, cache)
    return out
