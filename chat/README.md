# Alta Chat

A GTK 4 / libadwaita chat window for the 8-bit LLM engine (SmolLM2 135M / 1.7B Instruct) on the A311D.

- **NPU mode:** the prompt runs as a 256-token chunk on the NPU. The reply is generated on the CPU, straight from the NPU's own coefficient buffers (mesa-llm `etna_ml_llm_coefs`). The engine starts through `scripts/npu-attempt` → `npu-run`, so it gets the NPU lock, the kernel-error watch, the STOP latch and a PROGRESS.md record. It ends itself after 3 h (`--max-s`), 30 min before npu-run's deadline.
- **CPU mode:** everything runs on the CPU (int16 decode, with no 8-bit activations). It needs no NPU.
- **Engine:** `npu-llm/bin/jb_prefill-4bfd0dc8 --serve SOCKET` (llm/l3/jb_prefill.c). It runs as the systemd user unit `alta-chat-engine` on `$XDG_RUNTIME_DIR/alta-chat/engine.sock`, so the model stays loaded if the window closes. The protocol is described above `chat_serve()`.
- **Context reuse:** each turn sends the whole conversation as token ids. The engine reuses its cached prefix and processes only the new tokens, either with CPU decode steps or with a fresh NPU chunk, whichever its running timings say is cheaper.
- **KV cache:** fp16 and head-major. The 135M's teacher-forced logits vs fp32: KL 5e-7.
- **NPU guard:** the window runs under `no-npu-guard.so` (see the launcher `alta-chat`), because panvk (Vulkan 1.3) probes every render node.

Install the menu entry and icon (~/.local/share only): `chat/install.sh`. Remove them: `chat/install.sh --remove`.

## Memory (1.7B)
The 1.7B needs about 2.5 GB of free memory; below that the board swaps and replies slow from ~0.3 s to ~3 s per token (CHAT-1p7b, 2026-11-24). The app warns once before loading, but doesn't refuse. Two settings cut what it needs:
- `ETNA_ML_SHARE_TENSORS=1` (mesa-llm): one set of NPU input and output buffers for all layers (llm/l3/share-ladder.sh).
- `JB_HEAD8=1` (1.7B only): an 8-bit copy of the word table. The bf16 table's pages are dropped, so each token reads 101 MB of memory instead of 201 MB of disk-backed pages. Measured on the float model's own hidden states: KL 5.6e-4 / 1.8e-3, perplexity 2.176 → 2.186.

The engine's threads are pinned to the four A73 cores (`GOMP_CPU_AFFINITY=2-5`).
