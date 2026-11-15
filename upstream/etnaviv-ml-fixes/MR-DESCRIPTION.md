etnaviv/ml: V7 int8, addition and layout fixes

Seven fixes for the etnaviv NPU driver, found on a VIPNano-QI.7120 (Amlogic A311D, Libre Computer Alta) while
running small synthetic models through Teflon and comparing every output with the CPU interpreter.

1. **Detranspose outputs with a single row or column.** `is_3d()` skipped `[1,1,W,C]` and `[1,H,1,C]` graph
   outputs (the shape a `[1,T,C]` transformer tensor has), which stayed in NCHW order.
2. **Don't use a tile wider than the image on V7.** With H > W and W < 64 the NN job never completed
   (kernel hang report, FE past the NN trigger, core busy).
3. **Pad the kernels of the last NN core on V7.** Out-of-bounds read of the weight and bias tensors when the
   output channels don't divide evenly between the cores.
4. **Shift signed weights in the V7 coefficient stream.** int8 weights were encoded unshifted next to the
   shifted zero point; every int8 convolution was wrong on V7.
5. **Use the shifted zero points for the bias of V7 additions.**
6. **Don't read past the weights of an addition on V7.** The bias correction read 2 * 2 * input_channels bytes
   of the 8-byte addition kernel; the bias depended on heap contents, for uint8 too.
7. **Bring the operands of an addition to the same layout.** NHWC graph input + NCHW convolution output were
   added byte by byte, and the output wasn't detransposed.

### Testing

Hardware: VIPNano-QI.7120, kernel 6.12 with etnaviv. Tested on hardware with this branch (main 8a9009e0482 plus
the seven commits) and, before that, with the same changes in a tree based on main from April 2026; both give
byte-identical outputs:

- int8 1x1, 1x1 + RELU, 3x3, strided 3x3 and depthwise convolutions, and uint8/int8 `ADD(x, conv(x))`:
  within 1 level of the CPU; each int8 output is bit-identical to the output of the equivalent uint8 model
  (all values and zero points shifted by 128) minus 128.
- 1x1 convolutions on `[1,1,64,384]`, `[1,64,1,384]` and `[1,64,8,64]`: within 1 level of the CPU (before:
  the first was left in NCHW order and the second hung; the third was only run with the fix).
- MobileNetV1 (uint8) and 15 other uint8 models: configurations and coefficient buffers unchanged.

This branch (main + the seven commits) builds, and with a DRM shim that records submits instead of running
them it produces exactly the NN/TP configurations, coefficient buffers and command streams of the tree that
was tested on hardware, for 20 models. I have not run the Teflon
test suite. I have no V8 hardware; patches 1 and 7 touch the layout pass that V8 shares.

Not addressed: per-axis quantized convolutions are still rejected.

The code was generated with an LLM (see the `Generated-by: LLM` trailers); I have reviewed the patches.

<!-- Jordan: Mesa requires merge request text in your own words. Rewrite this file before pasting it. -->
