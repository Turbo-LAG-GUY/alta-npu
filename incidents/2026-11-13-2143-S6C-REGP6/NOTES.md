# Incident 2026-11-13 21:43 - S6C-REGP6 (register budget probe, boot b8c53cd6)

- Job: research/port/step6c-ladder.sh rung 2 = npu-attempt S6C-REGP6 = npu-shader/cs_run --kernel regp6 -n 256 -w 64
  with ETNA_CS_BARRIER_REG_BUDGET=576 (Jordan's blanket OK for the step 6c ladder; Jordan rebooted and cleared STOP
  after incident 2026-11-13-2109-S6C-SM1001 and said to continue). regp6 = the proven reverse kernel (one barrier)
  plus 6 vec4 values live across the barrier: 57 instructions, 9 temps programmed, 64 threads -> 576 registers.
- Result: "recover hung GPU!" once, no MMU fault, output untouched (0xdeadbeef). npu-run rc=90, STOP set.
  Board up. Ladder stopped (softmax and MobileNet rungs not run).
- Rung 1 just before, S6C-REGP4 (same kernel, 4 vec4 live, 7 temps, 64 x 7 = 448): PASS.
- Conclusion: a 64-thread workgroup with a barrier hangs at 9 programmed temps (576) and runs at 7 (448), with the
  same signature as the 16-temp softmax (incident 2026-11-13-2109-S6C-SM1001). The register-capacity explanation
  holds; the limit is between 448 and 576 (8 temps = 512 untested).
- Changes after this (CPU only): driver default budget 448 (largest proven); softmax picks the largest workgroup
  (64, 32, 16 threads) whose threads x temps fits the budget; ladder script without the 576 probe.
- Next: reboot (npu-run refuses this boot), clear runs/STOP after review, then the revised ladder.

## Follow-up 2026-11-13 22:12 (boot 8f7f01ab, after Jordan's reboot)
- S6C-REGP5 (64 x 8 = 512 registers, Jordan: "test it again at 512"): PASS. So the barrier-workgroup limit is
  between 512 and 576; the driver's default budget is now 512.
- The 32-thread softmax (352 registers) passed alone (1001/1001 and 4000/4000 exact) and inside MobileNet
  (S6C-MN-SM, S6C-MN-SM50: outputs byte-identical to the switch-off run).
