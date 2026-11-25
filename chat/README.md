# Alta Chat

A GTK 4 / libadwaita chat window for the 8-bit LLM engine (SmolLM2 135M / 1.7B Instruct) on the A311D.

- **NPU mode:** the prompt runs as a 256-token chunk on the NPU. The reply is generated on the CPU, straight from the NPU's own coefficient buffers (mesa-llm `etna_ml_llm_coefs`). The engine starts through `scripts/npu-attempt` → `npu-run`, so it gets the NPU lock, the kernel-error watch, the STOP latch and a PROGRESS.md record. It ends itself after 3 h (`--max-s`), 30 min before npu-run's deadline.
- **CPU mode:** everything runs on the CPU (int16 decode, with no 8-bit activations). It needs no NPU.
- **Engine:** `npu-llm/bin/jb_prefill-21e4e4e9 --serve SOCKET` (llm/l3/jb_prefill.c). It runs as the systemd user unit `alta-chat-engine` on `$XDG_RUNTIME_DIR/alta-chat/engine.sock`, so the model stays loaded if the window closes. The protocol is described above `chat_serve()`.
- **Context reuse:** each turn sends the whole conversation as token ids. The engine reuses its cached prefix and processes only the new tokens, either with CPU decode steps or with a fresh NPU chunk, whichever its running timings say is cheaper.
- **KV cache:** fp16 and head-major. The 135M's teacher-forced logits vs fp32: KL 5e-7.
- **NPU guard:** the window runs under `no-npu-guard.so` (see the launcher `alta-chat`), because panvk (Vulkan 1.3) probes every render node.

Install the menu entry and icon (~/.local/share only): `chat/install.sh`. Remove them: `chat/install.sh --remove`.
