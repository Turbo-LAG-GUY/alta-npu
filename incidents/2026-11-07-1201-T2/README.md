# Incident 2026-11-07 12:01 - T2 MMU faults

What ran: one npu-run job (T2): npu-classify.py, pinned coral mobilenet_v1_1.0_224_quant.tflite
(sha256 2d0ebfa2...), grace_hopper.bmp, Teflon from Mesa 26.1.0-devel c5423d3e273, -n 5, TEFLON_DEBUG=verbose.

What happened: every invoke -> "etnaviv-gpu ff100000.npu: MMU 0 fault (slave not present) addr 0x1f238600"
then "recover hung GPU!" (5 times, ~48 ms apart, kernel uptime 28536.44-28536.65 s). Also
"failed to allocate devcoredump file" (no devcoredump exists). The process exited rc 0 after 4 s with a
garbage result (top-5 all 0.0039). Teflon split the model into a 27-op and a 2-op graph. The board stayed
responsive; the NPU returned to runtime-suspended; no display (CREATE_DUMB) errors.

Model note: the Mesa-docs copy ~/Documents/AI_Projects/mesa/mobilenet_v1_1.0_224_quant.tflite
(sha256 1ce11b7e...) ends CONV-SQUEEZE-SOFTMAX with input scale 1/127.5. This pinned copy ends
CONV-RESHAPE-SOFTMAX with input scale 1/128. The Mesa-docs copy ran ~360 inferences with no kernel
errors on 2026-11-07 01:17 (same kernel, module and Teflon build).

Files: kernel-thisboot.txt, 20261107-120136-T2/ (meta, stdout, stderr, kernel.txt), observe log (runs/T2-observe.txt).
Per the handoff: reboot before any further NPU work; Jordan reviews, then rm ~/alta-npu/runs/STOP.
