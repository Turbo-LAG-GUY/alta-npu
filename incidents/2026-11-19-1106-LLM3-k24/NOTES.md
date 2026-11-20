# 2026-11-19 11:06 - NPU hang: LLM-L3 ladder, 1x1 conv with 24 kernels per core (driver cap 21)

- Run on purpose: LLM-engine step L3, `llm/l3/llm-l3-ladder.sh`, Jordan's OK ("You have the OK for the L3 ladder").
  Library: mesa-llm (branch llm-tiling-experiments, lib sha 678775c5) = tested series lib b6c2a28d plus the test
  hooks ETNA_ML_V7_KPC_CAP / ETNA_ML_V7_TILE_H (patch llm/l3/0001-etnaviv-ml-v7-tiling-experiment-hooks-LLM-engine-L3-.patch).
  Unit alta-llm-l3 (systemd-run --user), boot ea2b8d2f, kernel 6.12.73, keep_powered, AXI SRAM on, Steam not running.
- Rungs 1-4 (no override) PASS, max diff 0/1 vs CPU: smol_gateup 1x1 0.86 ms, 4x4 3.36 ms, 16x16 28.82 ms,
  qwen_gateup 1x1 1.65 ms (L1c: 0.66 / 3.29 / 31.17 / 1.62).
- Rung 5 HUNG: smol_gateup_1x1_P (576 -> 3072 int8 1x1 conv, one token), ETNA_ML_V7_KPC_CAP=24, TILE_H=0.
  "recover hung GPU!" at uptime 37736.30 and again at 37737.42, offending task = teflon_check.py; npu-run rc 90,
  STOP latch set, board stayed up, kernel recovered.
- The only NN-config difference from the passing default (offline dumps in `offline/`): kernels_per_core 21 -> 24
  (and the weight stream laid out for 24). Tile 1x1, interleave 1, everything else identical.
- Meaning: the driver's cap of accum_buffer_depth / 3 = 64 / 3 = 21 kernels per core for 1x1 kernels is a real
  hardware limit (or at least 24 is past it), not a conservative choice. The accumulation-buffer bound the hook kept
  (depth * interleave / tile_y = 64) is not the binding one. Every remaining rung raised kpc above 21 -> all dropped.
- devcoredump (saved by alta-devcd-watch, no password): `devcoredump.bin` (1868008 bytes). Decoded with
  research/port/tools/devcoredump_decode.py: FE DMA in the RING at offset 0xb80 (LINK/WAIT loop),
  HI_IDLE_STATE 0x7ffffff6 = all idle except FE (bit 0) and bit 3: the NN job never finished.
- Files: `ladder/` (rung outputs), `run/` (npu-run meta, kernel.txt), `capture/` (observer), `offline/` (fake-NPU
  config dumps for kpc 21 and 24).
- Next NPU job needs: a reboot (npu-run refuses after a hang in this boot), Jordan's review, `rm ~/alta-npu/runs/STOP`.
