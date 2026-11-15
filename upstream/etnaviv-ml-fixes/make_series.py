#!/usr/bin/env python3
"""make_series.py: apply the seven etnaviv/ml fixes to the upstream worktree (mesa-main, branch
alta-etnaviv-ml-fixes, based on mesa main 8a9009e0482 of 2026-11-14), one commit each. Run once on a clean worktree.
No Signed-off-by: Jordan adds it (git commit --amend -s / git rebase --signoff) when he sends the series."""
import subprocess
import sys

W = "/home/jordan/Documents/AI_Projects/mesa-upstream-fixes"
D = W + "/src/gallium/drivers/etnaviv/"
# Mesa docs/submittingpatches.rst: "Generated-by: LLM", no Co-authored-by for tools, no model name.
TRAILER = "\n\nGenerated-by: LLM\n"


def edit(name, old, new):
    p = D + name
    s = open(p).read()
    if s.count(old) != 1:
        sys.exit("%s: anchor found %d times:\n%s" % (name, s.count(old), old))
    open(p, "w").write(s.replace(old, new))


def commit(subject, body):
    subprocess.check_call(["git", "-C", W, "commit", "-q", "-a", "-m", subject + "\n\n" + body.strip() + TRAILER])


# 1 ---------------------------------------------------------------------------------------------------------------
edit("etnaviv_ml.c", """static bool
is_3d(struct pipe_tensor *tensor)
{
   return tensor->dims[1] > 1 &&
          tensor->dims[2] > 1 &&
          tensor->dims[3] > 1;
}
""", """/* Whether NCHW and NHWC put the bytes of this tensor in a different order:
 * only with more than one channel and more than one pixel.
 */
static bool
layouts_differ(struct pipe_tensor *tensor)
{
   return tensor->dims[3] > 1 &&
          tensor->dims[1] * tensor->dims[2] > 1;
}
""")
edit("etnaviv_ml.c", "             is_3d(poperation->output_tensors[i]) &&", "             layouts_differ(poperation->output_tensors[i]) &&")
edit("etnaviv_ml_tp.c", """   unsigned input_width = operation->input_width;
   unsigned input_height = operation->input_height;
   unsigned input_channels = operation->input_channels;
   struct etna_bo *bo = etna_ml_create_bo(subgraph->screen, sizeof(struct etna_tp_params));

   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_WRITE);

   struct etna_tp_params *map = etna_bo_map(bo);

   set_default_tp_config(map);

   map->in_image_x_size = input_width;
   map->in_image_y_size = input_height * input_channels;
   map->in_image_z_size = 0x1;
""", """   unsigned input_width = operation->input_width;
   unsigned input_height = operation->input_height;
   unsigned input_channels = operation->input_channels;

   /* The input is read as one linear run of planes, so 1xHxC and Hx1xC are
    * the same permutation. Use the second form: it has no x size of 1, and
    * its y size is C instead of H * C, which only has 16 bits.
    */
   if (input_width == 1 && input_height > 1) {
      input_width = input_height;
      input_height = 1;
   }

   struct etna_bo *bo = etna_ml_create_bo(subgraph->screen, sizeof(struct etna_tp_params));

   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_WRITE);

   struct etna_tp_params *map = etna_bo_map(bo);

   set_default_tp_config(map);

   map->in_image_x_size = input_width;
   map->in_image_y_size = input_height * input_channels;
   map->in_image_z_size = 0x1;
""")
commit("etnaviv/ml: Detranspose outputs with a single row or column", """
A graph output is only brought back from NCHW to NHWC when is_3d() holds,
that is with more than one row, column and channel. But the two layouts
also differ for 1xWxC and Hx1xC tensors (TFLite hands a [1, T, C]
transformer tensor to the delegate as [1, 1, T, C]), so the output of a
convolution on such a tensor reached the application still in channel
planes.

Detranspose whenever the tensor has more than one channel and more than
one pixel. For the single-row case configure the TP as Hx1xC, which is
the same permutation: the 1xHxC form has an x size of 1 and puts H * C in
the 16-bit y size.

Tested in a local tree with the same change on a VIPNano-QI.7120
(Amlogic A311D): the output of a 1x1 convolution on [1, 1, 64, 384] was
left in planar order before, and is within 1 level of the CPU now. 3D tensors keep their TP
configuration bit for bit.

Fixes: c728e73d650 ("etnaviv/ml: Track memory layout of tensors")
""")

# 2 ---------------------------------------------------------------------------------------------------------------
edit("etnaviv_ml_nn_v7.c", """   if (operation->pooling_first_pixel) {
      output_width *= 2;
      output_height *= 2;
   }

   tile_width = MIN2(output_width, 64);
""", """   if (operation->pooling_first_pixel) {
      output_width *= 2;
      output_height *= 2;
   }

   /* The image is configured with x = height and y = width (see
    * create_nn_config()), while the tile is computed from the unswapped
    * sizes. When that gives a tile that is wider than the configured image,
    * the job never finishes, so tile the image as it is configured.
    */
   if (!operation->fully_connected && MIN2(output_width, 64) > output_height) {
      unsigned tmp = output_width;
      output_width = output_height;
      output_height = tmp;
   }

   tile_width = MIN2(output_width, 64);
""")
commit("etnaviv/ml: Don't use a tile wider than the image on V7", """
etna_ml_calculate_tiling_v7() computes the tile from the operation's
width and height, but the NN configuration swaps the two when it fills
in the image sizes. For square images and for H <= W that makes no
difference to the result. With H > W and W < 64 the tile can end up wider
than the configured image, in the worst case a 64x1 tile on a 1x64 image
for an Hx1xC tensor, and the NN job never completes: the kernel reports
a hang, with the FE past the NN trigger and the core still busy.

Compute the tile from the swapped sizes in just that case. All other
tilings stay as they were: the configurations and coefficient buffers of
MobileNetV1 and of 15 other models are bit-identical with and without
this change.

Tested in a local tree with the same change on a VIPNano-QI.7120
(Amlogic A311D): a 1x1 convolution on [1, 64, 1, 384] hung the NPU on
every run before. With this change that model and [1, 64, 8, 64] match
the CPU within 1 level.
""")

# 3 ---------------------------------------------------------------------------------------------------------------
edit("etnaviv_ml_nn_v7.c", """   unsigned max_core_size = 0;
   unsigned bo_size;

   bo_size = calculate_weight_bo_size(subgraph, operation);
""", """   unsigned max_core_size = 0;
   unsigned bo_size;
   unsigned total_kernels = DIV_ROUND_UP(output_channels, cores_used) * cores_used;
   struct etna_operation padded;

   /* When the kernels don't divide evenly between the cores, the writers
    * below read kernels and biases past the end of the tensors for the last
    * core. Give those unused kernels zero weights and biases.
    */
   if (total_kernels > output_channels) {
      size_t kernel_size = operation->weight_width * operation->weight_height * input_channels;

      memcpy(&padded, operation, sizeof(padded));
      padded.weight_tensor = malloc(total_kernels * kernel_size);
      memcpy(padded.weight_tensor, operation->weight_tensor, output_channels * kernel_size);
      memset(padded.weight_tensor + output_channels * kernel_size, operation->weight_zero_point,
             (total_kernels - output_channels) * kernel_size);
      padded.weight_tensor_size = total_kernels * kernel_size;

      padded.bias_tensor = calloc(total_kernels, sizeof(uint32_t));
      memcpy(padded.bias_tensor, operation->bias_tensor, output_channels * sizeof(uint32_t));
      padded.bias_tensor_size = total_kernels * sizeof(uint32_t);

      operation = &padded;
   }

   bo_size = calculate_weight_bo_size(subgraph, operation);
""")
edit("etnaviv_ml_nn_v7.c", """   *cache_size = max_core_size * cores_used;

   return compressed;
""", """   *cache_size = max_core_size * cores_used;

   if (operation == &padded) {
      free(padded.weight_tensor);
      free(padded.bias_tensor);
   }

   return compressed;
""")
commit("etnaviv/ml: Pad the kernels of the last NN core on V7", """
The V7 coefficient writers emit DIV_ROUND_UP(output_channels, cores) *
cores kernels. When the output channels don't divide evenly between the
NN cores, the kernels and biases of the last core are read from past the
end of the weight and bias tensors: an out-of-bounds heap read, whose
contents end up in the coefficient buffer.

Pad the tensors with zero-valued kernels and biases for those unused
slots. Buffers of operations whose kernels divide evenly are unchanged.
""")

# 4 ---------------------------------------------------------------------------------------------------------------
edit("etnaviv_ml_nn_v7.c", """   /* When the kernels don't divide evenly between the cores, the writers
    * below read kernels and biases past the end of the tensors for the last
    * core. Give those unused kernels zero weights and biases.
    */
   if (total_kernels > output_channels) {
      size_t kernel_size = operation->weight_width * operation->weight_height * input_channels;

      memcpy(&padded, operation, sizeof(padded));
      padded.weight_tensor = malloc(total_kernels * kernel_size);
      memcpy(padded.weight_tensor, operation->weight_tensor, output_channels * kernel_size);
""", """   /* When the kernels don't divide evenly between the cores, the writers
    * below read kernels and biases past the end of the tensors for the last
    * core. Give those unused kernels zero weights and biases.
    *
    * Signed weights are still int8 here, while weight_zero_point is already
    * in the shifted 0..255 range (see etna_tensor_zero_point()). The writers
    * compare the weights with it and use both for the bias correction, so
    * shift the weights as well, as the V8 encoder does.
    */
   if (total_kernels > output_channels || operation->weight_signed) {
      size_t kernel_size = operation->weight_width * operation->weight_height * input_channels;

      memcpy(&padded, operation, sizeof(padded));
      padded.weight_tensor = malloc(total_kernels * kernel_size);
      memcpy(padded.weight_tensor, operation->weight_tensor, output_channels * kernel_size);
      if (operation->weight_signed) {
         for (size_t i = 0; i < output_channels * kernel_size; i++)
            padded.weight_tensor[i] ^= 0x80;
         padded.weight_signed = false;
      }
""")
commit("etnaviv/ml: Shift signed weights in the V7 coefficient stream", """
Signed tensors are stored shifted by 128 (see etna_tensor_zero_point()),
and the zero point of signed weights is shifted accordingly. The V8
encoder adds 128 to each signed weight when it encodes it, but the V7
writers put the int8 bytes in the stream as they are, next to the
shifted zero point, and compute the bias correction from the same
mismatched pair. A weight of 0 with zero point 0 is thus seen by the
hardware as 0 - 128.

Every int8 convolution is wrong on V7 because of this: a 1x1 convolution
with identity weights returns 127 for every output.

Shift the weights before they are encoded. With this, an int8 model
produces exactly the configurations and coefficient buffers of the uint8
model that has all its values and zero points shifted by 128.

Tested in a local tree with the same change on a VIPNano-QI.7120 (Amlogic A311D) with int8 1x1, 3x3, strided
3x3 and depthwise convolutions: all within 1 level of the CPU, and
bit-identical to the outputs of the equivalent uint8 models minus 128.
""")

# 5 ---------------------------------------------------------------------------------------------------------------
edit("etnaviv_ml_nn.c", """   bias_map[0] = compute_bias_add(poperation->input_tensors[1]->scale, poperation->input_tensors[0]->scale,
                                  poperation->input_tensors[1]->zero_point, poperation->input_tensors[0]->zero_point,
                                  operation->weight_scale);
""", """   bias_map[0] = compute_bias_add(poperation->input_tensors[1]->scale, poperation->input_tensors[0]->scale,
                                  etna_tensor_zero_point(poperation->input_tensors[1]),
                                  etna_tensor_zero_point(poperation->input_tensors[0]),
                                  operation->weight_scale);
""")
commit("etnaviv/ml: Use the shifted zero points for the bias of V7 additions", """
etna_ml_lower_add_v7() passes the raw zero points of its inputs to
compute_bias_add(). With int8 inputs these are in -128..127: they wrap
in the uint8_t parameters, and the offset term is 128 levels off from
what the hardware needs for the stored (shifted) tensors.

Use etna_tensor_zero_point(), as for all other operands. Nothing changes
for uint8 inputs.
""")

# 6 ---------------------------------------------------------------------------------------------------------------
edit("etnaviv_ml_nn_v7.c", """   int32_t correction = 0;

   for (unsigned i = 0; i < operation->weight_width * operation->weight_height * operation->input_channels; i++) {
""", """   int32_t correction = 0;
   /* The weights of an addition are a single 2x2 kernel with one channel */
   unsigned input_channels = operation->addition ? 1 : operation->input_channels;

   for (unsigned i = 0; i < operation->weight_width * operation->weight_height * input_channels; i++) {
""")
commit("etnaviv/ml: Don't read past the weights of an addition on V7", """
The weight tensor of an addition is 8 bytes: a 2x2 kernel in which only
the first weight is set, which the coefficient writers treat as having a
single input channel. calculate_bias_correction() however sums
weight_width * weight_height * input_channels bytes, with the channel
count of the tensors being added: for 64 channels it reads 256 bytes
from an 8-byte allocation and adds whatever follows it to the bias.

The bias of an addition, uint8 or int8, thus depended on the contents of
the heap: compiling the same uint8 model three times gave three
different bias values.

Limit the sum to the kernel. The correction is then weight[0] *
input_zero_point, and the resulting bias is the expected
-addition_offset * zp1 - weight[0] * zp0.
""")

# 7 ---------------------------------------------------------------------------------------------------------------
edit("etnaviv_ml.c", """/** Tensor layout inference:""", """/* Whether one of the operands of an element-wise operation is in NCHW order.
 * Graph inputs are in NHWC order, and the layouts of the other operands are
 * known because their producers have been lowered already.
 */
static bool
any_input_is_nchw(struct etna_ml_subgraph *subgraph,
                  const struct pipe_ml_operation *poperations,
                  unsigned count,
                  const struct pipe_ml_operation *poperation)
{
   for (unsigned i = 0; i < poperation->input_count; i++) {
      struct pipe_tensor *tensor = poperation->input_tensors[i];

      if (tensor->data != NULL ||
          !etna_ml_find_producer(poperations, count, tensor->index))
         continue;

      if (etna_ml_get_tensor(subgraph, tensor->index)->act_layout == ETNA_ML_LAYOUT_NCHW)
         return true;
   }

   return false;
}

/** Tensor layout inference:""")
edit("etnaviv_ml.c", """  * - Element-wise operations don't care about the layout.
""", """  * - Element-wise operations don't care about the layout, as long as all their operands have the same one.
""")
edit("etnaviv_ml.c", """             poperation->type == PIPE_ML_OPERATION_TYPE_CONCATENATION)
            operation_layout = ETNA_ML_LAYOUT_NCHW;

         if (!etna_ml_find_producer(poperations, count, poperation->input_tensors[i]->index)) {""", """             poperation->type == PIPE_ML_OPERATION_TYPE_CONCATENATION)
            operation_layout = ETNA_ML_LAYOUT_NCHW;

         /* An element-wise operation needs its operands in the same order */
         if ((poperation->type == PIPE_ML_OPERATION_TYPE_ADD ||
              poperation->type == PIPE_ML_OPERATION_TYPE_SUBTRACT) &&
             layouts_differ(poperation->input_tensors[i]) &&
             any_input_is_nchw(subgraph, poperations, count, poperation))
            operation_layout = ETNA_ML_LAYOUT_NCHW;

         if (!etna_ml_find_producer(poperations, count, poperation->input_tensors[i]->index)) {""")
commit("etnaviv/ml: Bring the operands of an addition to the same layout", """
Additions and subtractions are lowered as layout-agnostic. That only
holds when both operands are in the same order. When a graph input
(NHWC) is added to the output of a convolution (NCHW), as in a residual
connection around the first convolution of a subgraph, the two buffers
are added byte by byte across the two layouts. The output also takes
the layout of the first operand, so if that is the NHWC one, the result
is never detransposed.

When any operand is in NCHW order, transpose the others to it. The
output then inherits NCHW and gets the usual detranspose at the graph
output. Operations whose operands already agree are lowered as before.

Tested in a local tree with the same change on a VIPNano-QI.7120 (Amlogic A311D) with x -> 1x1 convolution ->
y, ADD(x, y) on [1, 8, 8, 64], uint8 and int8 (together with the three
previous commits): up to 154 levels off the CPU before, and the output
was the CPU's formula applied to x in NHWC and y in NCHW order; within 1
level now.
""")
print("7 commits made")
