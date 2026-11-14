/*
 * Copyright (c) 2023-2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <unistd.h>
#include "pipe/p_state.h"
#include <sys/time.h>

#include "util/mesa-blake3.h"
#include "util/u_inlines.h"

#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_emit.h"
#include "etnaviv_ml_nn.h"
#include "etnaviv_ml_tp.h"
#include "etnaviv_ml.h"
#include "etnaviv_ml_shader.h"
#include "etnaviv_ml_axi_sram.h"
#include "etnaviv_ml_disk_cache.h"
#include "etnaviv_screen.h"

struct etna_ml_tensor *
etna_ml_get_tensor(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   struct etna_ml_tensor **tensor = util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor*, idx);
   if (*tensor == NULL)
      *tensor = calloc(1, sizeof(**tensor));
   return *tensor;
}

struct pipe_resource *
etna_ml_get_resource(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   return etna_ml_get_tensor(subgraph, idx)->resource;
}

unsigned
etna_ml_get_offset(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   return etna_ml_get_tensor(subgraph, idx)->offset;
}

unsigned
etna_ml_get_size(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   return etna_ml_get_tensor(subgraph, idx)->size;
}

static void
etna_ml_copy_layout(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned src_idx)
{
   struct etna_ml_tensor *dst = etna_ml_get_tensor(subgraph, idx);
   struct etna_ml_tensor *src = etna_ml_get_tensor(subgraph, src_idx);
   dst->exp_layout = src->exp_layout;
   dst->act_layout = src->act_layout;
}

unsigned
etna_ml_allocate_tensor(struct etna_ml_subgraph *subgraph)
{
   struct etna_ml_tensor **tensors = util_dynarray_grow(&subgraph->tensors, struct etna_ml_tensor *, 1);
   tensors[0] = calloc(1, sizeof(*tensors[0]));
   tensors[0]->resource = NULL;
   tensors[0]->offset = 0;
   tensors[0]->size = 0;

   return util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *) - 1;
}

void
etna_ml_create_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)
{
   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);

   assert(idx < util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *));
   assert(size > 0);

   struct pipe_resource *res = tensor->resource;

   if (res != NULL) {
      assert(size == tensor->size);
      return;
   }

   res = etna_ml_create_resource(&subgraph->screen->base, size);
   tensor->resource = res;
   tensor->size = size;

   ML_DBG("created resource %p for tensor %d with size %d\n", res, idx, size);
}

static void
etna_ml_destroy_tensor(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);

   pipe_resource_reference(&tensor->resource, NULL);
   tensor->offset = 0;
   tensor->size = 0;
}

struct etna_bo *
etna_ml_create_bo(struct etna_screen *screen, size_t size)
{
   struct etna_bo *bo = etna_bo_new(screen->dev,
                                    size,
                                    DRM_ETNA_GEM_CACHE_WC);

   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_WRITE);
   struct etna_nn_params *map = etna_bo_map(bo);
   memset(map, 0, size);
   etna_bo_cpu_fini(bo);

   return bo;
}

struct pipe_resource *
etna_ml_create_resource(struct pipe_screen *pscreen, size_t size)
{
   struct pipe_resource *res = pipe_buffer_create(pscreen, 0, PIPE_USAGE_DEFAULT, size);
   void *ptr = etna_bo_map(etna_buffer_resource(res)->bo);
   memset(ptr, 0, pipe_buffer_size(res));

   return res;
}

struct etna_core_npu_info *
etna_ml_get_core_info(struct etna_screen *screen)
{
   struct etna_core_info *info = etna_gpu_get_core_info(screen->npu);
   return &info->npu;
}

static bool
needs_reshuffle(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation)
{
   unsigned nn_core_version = subgraph->screen->specs.nn_core_version;
   bool has_stride = poperation->conv.stride_x > 1 || poperation->conv.stride_y > 1;
   bool pointwise = poperation->conv.pointwise;
   bool has_padding = poperation->conv.padding_top > 0 ||
                      poperation->conv.padding_bottom > 0 ||
                      poperation->conv.padding_left > 0 ||
                      poperation->conv.padding_right > 0;
   unsigned input_width = poperation->input_tensors[0]->dims[1];

   if (!has_stride)
      return false;

   if (nn_core_version < 8)
      return !(poperation->conv.depthwise && (input_width > 5 || input_width < 3)) && !pointwise;
   else {
      unsigned input_channels = poperation->input_tensors[0]->dims[3];

      if (poperation->conv.depthwise)
         return false;

      if (poperation->conv.pointwise && input_width >= 3 && input_channels > 1)
         return false;

      if (poperation->conv.pointwise && has_padding)
         return false;

      return true;
   }
}

static const struct pipe_ml_operation *
etna_ml_find_producer(const struct pipe_ml_operation *poperations,
                      unsigned count,
                      unsigned tensor_idx)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->output_count; j++)
         if (poperation->output_tensors[j]->index == tensor_idx)
            return poperation;
   }

   return NULL;
}

static const struct pipe_ml_operation *
etna_ml_find_consumer(const struct pipe_ml_operation *poperations,
                      unsigned count,
                      unsigned tensor_idx)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->input_count; j++)
         if (poperation->input_tensors[j]->index == tensor_idx)
            return poperation;
   }

   return NULL;
}

static void
reference_tensor_with_offset(struct etna_ml_subgraph *subgraph,
                             unsigned src_tensor,
                             unsigned dst_tensor,
                             unsigned offset,
                             unsigned size)
{
   struct etna_ml_tensor *src = etna_ml_get_tensor(subgraph, src_tensor);
   struct etna_ml_tensor *dst = etna_ml_get_tensor(subgraph, dst_tensor);
   struct pipe_resource *old_res = dst->resource;
   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);
   unsigned num_tensors = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
   ML_DBG("src_tensor %d (%x) dst_tensor %d offset %d size %d\n", src_tensor, etna_bo_gpu_va(etna_buffer_resource(src->resource)->bo), dst_tensor, offset, size);
   pipe_resource_reference(&dst->resource, src->resource);
   dst->offset = offset;
   dst->size = size;

   if (old_res) {
      for (int i = 0; i < num_tensors; i++) {
         if (etna_ml_get_resource(subgraph, i) == old_res) {
            pipe_resource_reference(&tensors[i]->resource, src->resource);
            tensors[i]->size = size;
            tensors[i]->offset = offset;
         }
      }
   }
}

static void
recreate_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)
{
   struct pipe_resource *old_res = etna_ml_get_resource(subgraph, idx);
   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);
   unsigned num_tensors = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
   struct pipe_resource *new_res;

   etna_ml_destroy_tensor(subgraph, idx);
   etna_ml_create_tensor(subgraph, idx, size);
   new_res = etna_ml_get_resource(subgraph, idx);

   if (old_res) {
      for (int i = 0; i < num_tensors; i++) {
         if (etna_ml_get_resource(subgraph, i) == old_res) {
            pipe_resource_reference(&tensors[i]->resource, new_res);
            tensors[i]->size = size;
         }
      }
   }
}

static void
dump_graph(struct list_head *etna_operations)
{
   ML_DBG("\n");
   ML_DBG("dumping intermediate graph: %d operations\n", list_length(etna_operations));

   ML_DBG("\n");
   ML_DBG("%3s %-4s %3s %3s  %s\n", "idx", "type", "in", "out", "operation type-specific");
   ML_DBG("================================================================================================\n");
   unsigned i = 0;
   list_for_each_entry(struct etna_operation, operation, etna_operations, link) {
      switch(operation->type) {
      case ETNA_JOB_TYPE_TP:
         ML_DBG("%3d %-4s %3d %3d",
                i, "TP", operation->input_tensors[0], operation->output_tensors[0]);
         break;
      case ETNA_JOB_TYPE_NN:
         ML_DBG("%3d %-4s %3d %3d in2: %3d",
                i, "NN", operation->input_tensors[0], operation->output_tensors[0], operation->input_tensors[1]);
         break;
      case ETNA_JOB_TYPE_CONCAT:
         ML_DBG("%3d %-4s %3d %3d in2: %3d",
                i, "CONC", operation->input_tensors[0], operation->output_tensors[0], operation->input_tensors[1]);
         break;
      case ETNA_JOB_TYPE_SPLIT:
         ML_DBG("%3d %-4s %3d %3d out2: %3d",
                i, "SPLIT", operation->input_tensors[0], operation->output_tensors[0], operation->output_tensors[1]);
         break;
      case ETNA_JOB_TYPE_BYPASS:
         ML_DBG("%3d %-4s %3d %3d",
                i, "BYPASS", operation->input_tensors[0], operation->output_tensors[0]);
         break;
      case ETNA_JOB_TYPE_SHADER:
         ML_DBG("%3d %-4s %3d %3d",
                i, "SH", operation->input_tensors[0], operation->output_tensors[0]);
         break;
      }
      ML_DBG("\n");
      i++;
   }
   ML_DBG("\n");
}

static void
add_bypass(const struct pipe_ml_operation *poperation, unsigned input_tensor,
           struct etna_operation *operation, struct list_head *etna_operations)
{
   operation->type = ETNA_JOB_TYPE_BYPASS;
   operation->input_count = 1;
   operation->input_tensors[0] = input_tensor;
   operation->input_tensor_sizes[0] = poperation->input_tensors[0]->dims[1] *
                                      poperation->input_tensors[0]->dims[2] *
                                      poperation->input_tensors[0]->dims[3];
   operation->output_count = 1;
   operation->output_tensors[0] = poperation->output_tensors[0]->index;
   operation->output_tensor_sizes[0] = poperation->output_tensors[0]->dims[1] *
                                       poperation->output_tensors[0]->dims[2] *
                                       poperation->output_tensors[0]->dims[3];

   list_addtail(&operation->link, etna_operations);
}

static bool
is_3d(struct pipe_tensor *tensor)
{
   return tensor->dims[1] > 1 &&
          tensor->dims[2] > 1 &&
          tensor->dims[3] > 1;
}

/** Tensor layout inference:
  * - Graph inputs are in NHWC order.
  * - Graph outputs are expected in NHWC order.
  * - Element-wise operations don't care about the layout.
  * - Other operations expect the tensors in NCHW order (if input_channels > 1) and their outputs are in NCHW order.
  * - Implicit transposes and detransposes are the only operations that change channel order.
  * - Explicit transposes and detransposes are ignored.
  */

static void
lower_operations(struct etna_ml_subgraph *subgraph,
                 const struct pipe_ml_operation *poperations,
                 unsigned count,
                 struct list_head *etna_operations)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];
      struct etna_operation *operation = calloc(1, sizeof(*operation));
      assert(poperation->input_count <= MAX_TENSORS);
      unsigned input_tensors[MAX_TENSORS] = {};

      for (int i = 0; i < poperation->input_count; i++) {
         struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, poperation->input_tensors[i]->index);
         enum etna_ml_tensor_layout operation_layout = ETNA_ML_LAYOUT_ANY;

         /* the shader-core average pool reads channel planes too */
         if (poperation->type == PIPE_ML_OPERATION_TYPE_CONVOLUTION ||
             poperation->type == PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED ||
             poperation->type == PIPE_ML_OPERATION_TYPE_CONCATENATION ||
             poperation->type == PIPE_ML_OPERATION_TYPE_POOLING)
            operation_layout = ETNA_ML_LAYOUT_NCHW;

         if (!etna_ml_find_producer(poperations, count, poperation->input_tensors[i]->index)) {
            /* In TensorFlow Lite, graph inputs are in channel-last */
            tensor->act_layout = ETNA_ML_LAYOUT_NHWC;
            tensor->exp_layout = ETNA_ML_LAYOUT_NHWC;
         }

         input_tensors[i] = poperation->input_tensors[i]->index;

         if (poperation->input_tensors[i]->data != NULL)
            continue;

         if (operation_layout != ETNA_ML_LAYOUT_ANY &&
             tensor->act_layout != operation_layout) {
            ML_DBG("Adding transpose.\n");
            struct etna_operation *transpose = calloc(1, sizeof(*transpose));
            etna_ml_lower_transpose(subgraph, poperation->input_tensors[i], transpose);
            if (etna_ml_shader_wants_transpose(subgraph, transpose))
               transpose->type = ETNA_JOB_TYPE_SHADER;
            transpose->input_tensors[0] = input_tensors[i];
            transpose->output_tensors[0] = etna_ml_allocate_tensor(subgraph);
            input_tensors[i] = transpose->output_tensors[0];
            list_addtail(&transpose->link, etna_operations);

            struct etna_ml_tensor *transposed_tensor = etna_ml_get_tensor(subgraph, input_tensors[i]);
            transposed_tensor->exp_layout = tensor->exp_layout;
            transposed_tensor->act_layout = operation_layout;
         }

         struct etna_ml_tensor *tensor2 = etna_ml_get_tensor(subgraph, input_tensors[i]);
         ML_DBG("operation %d input tensor %d layouts %d %d.\n", poperation->type, input_tensors[i], tensor2->exp_layout, tensor2->act_layout);
      }

      switch(poperation->type) {
         case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
            if (needs_reshuffle(subgraph, poperation)) {
               ML_DBG("Adding reshuffle for convolution operation.\n");
               struct etna_operation *reshuffle = calloc(1, sizeof(*reshuffle));
               etna_ml_lower_reshuffle(subgraph, poperation, reshuffle);
               reshuffle->input_tensors[0] = input_tensors[0];
               reshuffle->output_tensors[0] = etna_ml_allocate_tensor(subgraph);
               input_tensors[0] = reshuffle->output_tensors[0];
               list_addtail(&reshuffle->link, etna_operations);
               etna_ml_copy_layout(subgraph, reshuffle->output_tensors[0], reshuffle->input_tensors[0]);
            }

            ML_DBG("Adding convolution.\n");
            etna_ml_lower_convolution(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            input_tensors[0] = operation->output_tensors[0];
            list_addtail(&operation->link, etna_operations);
            etna_ml_copy_layout(subgraph, operation->output_tensors[0], operation->input_tensors[0]);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_ADD: {
            etna_ml_lower_add(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->input_tensors[1] = input_tensors[1];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_CONCATENATION: {
            operation->type = ETNA_JOB_TYPE_CONCAT;
            assert(poperation->input_count <= MAX_TENSORS);
            for (int i = 0; i < poperation->input_count; i++) {
               operation->input_tensors[i] = input_tensors[i];
               operation->input_tensor_sizes[i] = poperation->input_tensors[i]->dims[1] *
                                                  poperation->input_tensors[i]->dims[2] *
                                                  poperation->input_tensors[i]->dims[3];
            }
            operation->input_count = poperation->input_count;

            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            operation->output_width = poperation->output_tensors[0]->dims[1];
            operation->output_height = poperation->output_tensors[0]->dims[2];
            operation->output_channels = poperation->output_tensors[0]->dims[3];
            operation->output_tensor_sizes[0] = operation->output_width *
                                                operation->output_height *
                                                operation->output_channels;

            list_addtail(&operation->link, etna_operations);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_SPLIT: {
            operation->type = ETNA_JOB_TYPE_SPLIT;

            operation->input_tensors[0] = poperation->input_tensors[1]->index;
            operation->input_tensor_sizes[0] = poperation->input_tensors[1]->dims[1] *
                                               poperation->input_tensors[1]->dims[2] *
                                               poperation->input_tensors[1]->dims[3];

            assert(poperation->output_count <= MAX_TENSORS);
            for (int i = 0; i < poperation->output_count; i++) {
               operation->output_tensors[i] = poperation->output_tensors[i]->index;
               operation->output_tensor_sizes[i] = poperation->output_tensors[i]->dims[1] *
                                                   poperation->output_tensors[i]->dims[2] *
                                                   poperation->output_tensors[i]->dims[3];
            }
            operation->output_count = poperation->output_count;

            list_addtail(&operation->link, etna_operations);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_PAD: {
            ML_DBG("Adding pad operation.\n");
            etna_ml_lower_pad(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED: {
            etna_ml_lower_fully_connected(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_RESHAPE: {
            add_bypass(poperation, input_tensors[0], operation, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_SOFTMAX: {
            etna_ml_lower_softmax(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_POOLING: {
            /* only global average pools are supported (etna_ml_shader_avgpool_supported) */
            etna_ml_lower_avgpool(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_RELU: {
            etna_ml_lower_relu(subgraph, poperation, poperation->input_tensors[0], operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_ABSOLUTE: {
            etna_ml_lower_absolute(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_LOGISTIC: {
            etna_ml_lower_logistic(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_SUBTRACT: {
            struct etna_operation *operation = calloc(1, sizeof(*operation));
            etna_ml_lower_add(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->input_tensors[1] = input_tensors[1];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
            add_bypass(poperation, input_tensors[0], operation, etna_operations);
            struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, operation->output_tensors[0]);
            tensor->act_layout = ETNA_ML_LAYOUT_NCHW;
            tensor->exp_layout = ETNA_ML_LAYOUT_NCHW;

            break;
         }
         default:
            UNREACHABLE("Unsupported ML operation type");
      }

      for (int i = 0; i < poperation->output_count; i++) {
         struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, poperation->output_tensors[i]->index);
         if (tensor->exp_layout == ETNA_ML_LAYOUT_ANY &&
             tensor->act_layout == ETNA_ML_LAYOUT_ANY) {
            ML_DBG("Copying layout to output tensor %d.\n", poperation->output_tensors[i]->index);
            etna_ml_copy_layout(subgraph, poperation->output_tensors[i]->index, operation->input_tensors[0]);
         }

         ML_DBG("type %d i %d tensor %d layout %d == %d\n", poperation->type, i, poperation->output_tensors[i]->index, tensor->exp_layout, tensor->act_layout);
         if (!etna_ml_find_consumer(poperations, count, poperation->output_tensors[i]->index) &&
             is_3d(poperation->output_tensors[i]) &&
             tensor->exp_layout != tensor->act_layout) {
            ML_DBG("Adding detranspose.\n");
            struct etna_operation *detranspose = calloc(1, sizeof(*detranspose));
            etna_ml_lower_detranspose(subgraph, poperation->output_tensors[i], detranspose);
            if (etna_ml_shader_wants_transpose(subgraph, detranspose))
               detranspose->type = ETNA_JOB_TYPE_SHADER;
            operation->output_tensors[i] = etna_ml_allocate_tensor(subgraph);
            detranspose->input_tensors[0] = operation->output_tensors[i];
            detranspose->output_tensors[0] = poperation->output_tensors[i]->index;
            list_addtail(&detranspose->link, etna_operations);
         }
      }
   }

   list_for_each_entry(struct etna_operation, operation, etna_operations, link) {
      if (operation->type == ETNA_JOB_TYPE_CONCAT) {
         etna_ml_create_tensor(subgraph, operation->output_tensors[0], operation->output_tensor_sizes[0]);

         unsigned offset = 0;
         for (int i = 0; i < operation->input_count; i++) {
            reference_tensor_with_offset(subgraph,
                                       operation->output_tensors[0],
                                       operation->input_tensors[i],
                                       offset,
                                       operation->input_tensor_sizes[i]);
            offset += operation->input_tensor_sizes[i];
         }
      } else if (operation->type == ETNA_JOB_TYPE_SPLIT) {
         etna_ml_create_tensor(subgraph, operation->input_tensors[0], operation->input_tensor_sizes[0]);

         unsigned offset = 0;
         for (int i = 0; i < operation->output_count; i++) {
            reference_tensor_with_offset(subgraph,
                                         operation->input_tensors[0],
                                         operation->output_tensors[i],
                                         offset,
                                         operation->output_tensor_sizes[i]);
            offset += operation->output_tensor_sizes[i];
         }
      } else if (operation->type == ETNA_JOB_TYPE_NN && operation->input_count > 1) { /* Add or Subtraction */
         recreate_tensor(subgraph, operation->input_tensors[0], operation->input_tensor_sizes[0] +
                                                                        operation->input_tensor_sizes[1]);
         reference_tensor_with_offset(subgraph,
                                      operation->input_tensors[0],
                                      operation->input_tensors[1],
                                      operation->input_tensor_sizes[0],
                                      operation->input_tensor_sizes[1]);
      } else if (operation->type == ETNA_JOB_TYPE_BYPASS) {
         etna_ml_create_tensor(subgraph, operation->input_tensors[0], operation->input_tensor_sizes[0]);
         reference_tensor_with_offset(subgraph,
                                      operation->input_tensors[0],
                                      operation->output_tensors[0],
                                      0,
                                      operation->output_tensor_sizes[0]);
      } else {
         for (int i = 0; i < operation->input_count; i++)
            etna_ml_create_tensor(subgraph, operation->input_tensors[i], operation->input_tensor_sizes[i]);
      }
   }

   /* Create any output tensors that aren't inputs to other operations, these
    * are the outputs of the graph.
    */
   ML_DBG("Ensuring all output tensors have their memory backing.\n");
   list_for_each_entry(struct etna_operation, operation, etna_operations, link) {
      struct pipe_resource *res = etna_ml_get_resource(subgraph, operation->output_tensors[0]);
      if (res != NULL)
         continue;

      etna_ml_create_tensor(subgraph, operation->output_tensors[0], operation->output_tensor_sizes[0]);
   }

   if (DBG_ENABLED(ETNA_DBG_ML_MSGS))
      dump_graph(etna_operations);
}

static unsigned
count_tensors(const struct pipe_ml_operation *poperations,
              unsigned count)
{
   unsigned tensor_count = 0;

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->input_count; j++)
         tensor_count = MAX2(tensor_count, poperation->input_tensors[j]->index);

      for (unsigned j = 0; j < poperation->output_count; j++)
         tensor_count = MAX2(tensor_count, poperation->output_tensors[j]->index);

      switch (poperation->type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION:
         tensor_count = MAX2(tensor_count, poperation->conv.weight_tensor->index);
         tensor_count = MAX2(tensor_count, poperation->conv.bias_tensor->index);
         break;
      case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED:
         tensor_count = MAX2(tensor_count, poperation->fcon.weight_tensor->index);
         tensor_count = MAX2(tensor_count, poperation->fcon.bias_tensor->index);
         break;
      case PIPE_ML_OPERATION_TYPE_PAD:
      case PIPE_ML_OPERATION_TYPE_SUBTRACT:
      case PIPE_ML_OPERATION_TYPE_ADD:
      case PIPE_ML_OPERATION_TYPE_CONCATENATION:
      case PIPE_ML_OPERATION_TYPE_SPLIT:
      case PIPE_ML_OPERATION_TYPE_RESHAPE:
      case PIPE_ML_OPERATION_TYPE_RELU:
      case PIPE_ML_OPERATION_TYPE_ABSOLUTE:
      case PIPE_ML_OPERATION_TYPE_LOGISTIC:
      case PIPE_ML_OPERATION_TYPE_TRANSPOSE:
      case PIPE_ML_OPERATION_TYPE_SOFTMAX:
      case PIPE_ML_OPERATION_TYPE_POOLING:
         break;
      default:
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   return tensor_count + 1;
}

static bool
tensor_quantization_supported(struct pipe_tensor *tensor)
{
   /*
    * Per-axis quantization not supported, for details see:
    * https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor
    */
   return tensor->scales == NULL && tensor->zero_points == NULL;
}

bool
etna_ml_operation_supported(struct pipe_ml_device *pdevice,
                            const struct pipe_ml_operation *operation)
{
   bool supported = false;

   switch (operation->type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
         struct pipe_tensor *input_tensor = operation->input_tensors[0];
         struct pipe_tensor *weight_tensor = operation->conv.weight_tensor;
         struct pipe_tensor *bias_tensor = operation->conv.bias_tensor;
         struct pipe_tensor *output_tensor = operation->output_tensors[0];

         // Dilation and per-axis quantization not yet implemented
         if (tensor_quantization_supported(input_tensor) &&
             tensor_quantization_supported(weight_tensor) &&
             tensor_quantization_supported(bias_tensor) &&
             tensor_quantization_supported(output_tensor) &&
             operation->conv.dilation_width_factor == 1 &&
             operation->conv.dilation_height_factor == 1) {
            supported = true;
         }
         break;
      }
      case PIPE_ML_OPERATION_TYPE_SUBTRACT:
      case PIPE_ML_OPERATION_TYPE_ADD: {
         supported = operation->input_tensors[0]->data == NULL &&
                     operation->input_tensors[1]->data == NULL;
         break;
      }
      case PIPE_ML_OPERATION_TYPE_CONCATENATION: {
         supported = true;

         if (operation->conc.axis != 3 &&
             operation->conc.axis != -1)
            supported = false;

         break;
      }
      case PIPE_ML_OPERATION_TYPE_SPLIT: {
         supported = true;

         if (operation->conc.axis != 3 &&
             operation->conc.axis != -1)
            supported = false;

         unsigned output_channels = operation->output_tensors[0]->dims[3];
         for (unsigned i = 1; i < operation->output_count; i++)
            if (output_channels != operation->output_tensors[i]->dims[3])
               supported = false;

         break;
      }
      case PIPE_ML_OPERATION_TYPE_PAD: {
         supported = operation->pad.before_x <= 2 &&
                     operation->pad.after_x <= 2 &&
                     operation->pad.before_y <= 2 &&
                     operation->pad.after_y <= 2 &&
                     operation->pad.before_z <= 2 &&
                     operation->pad.after_z <= 2;
         break;
      }
      case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED: {
         supported = operation->input_tensors[0]->dims[3] < 1280;
         break;
      }
      case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
         if (operation->transpose.perm[0] == 0 &&
             operation->transpose.perm[1] == 3 &&
             operation->transpose.perm[2] == 1 &&
             operation->transpose.perm[3] == 2)
            supported = true;  /* Channels-last to channels-first */
         if (operation->transpose.perm[0] == 0 &&
             operation->transpose.perm[1] == 2 &&
             operation->transpose.perm[2] == 3 &&
             operation->transpose.perm[3] == 1)
            supported = true;  /* Channels-first to channels-last */
         break;
      }
      case PIPE_ML_OPERATION_TYPE_RESHAPE:
         /* a squeeze only where it lets a following shader-core op (softmax) join the graph */
         supported = !operation->reshape.squeeze ||
                     etna_ml_shader_softmax_enabled(etna_ml_device_screen(pdevice));
         break;
      case PIPE_ML_OPERATION_TYPE_RELU:
      case PIPE_ML_OPERATION_TYPE_ABSOLUTE:
      case PIPE_ML_OPERATION_TYPE_LOGISTIC:
         supported = true;
         break;
      case PIPE_ML_OPERATION_TYPE_SOFTMAX:
         supported = etna_ml_shader_softmax_supported(etna_ml_device_screen(pdevice), operation);
         break;
      case PIPE_ML_OPERATION_TYPE_POOLING:
         supported = etna_ml_shader_avgpool_supported(etna_ml_device_screen(pdevice), operation);
         break;
      default:
         return false;
   }

   return supported;
}

/*
 * alta: on-disk cache of compressed weights. Compressing them is most of what
 * compiling a model costs, and the result depends only on the model, the chip
 * and the code that compiles weights. On by default in this build. ETNA_ML_DISK_CACHE=0 turns
 * it off, =verbose reports its use, and =verify also recompiles every hit and
 * aborts on any difference, before anything reaches the NPU.
 */
enum ml_cache_mode { ML_CACHE_OFF, ML_CACHE_ON, ML_CACHE_VERBOSE, ML_CACHE_VERIFY };

struct ml_cache_blob_header {
   uint32_t bo_size;
   uint32_t cache_size;
   uint32_t data_size;
   uint32_t reserved;
};

static struct {
   unsigned hits, misses, stored;
} ml_cache_counts;

static enum ml_cache_mode
ml_cache_mode(void)
{
   static int mode = -1;

   if (mode < 0) {
      const char *env = getenv("ETNA_ML_DISK_CACHE");

      if (env && (!strcmp(env, "0") || !strcmp(env, "off")))
         mode = ML_CACHE_OFF;
      else if (env && !strcmp(env, "verbose"))
         mode = ML_CACHE_VERBOSE;
      else if (env && !strcmp(env, "verify"))
         mode = ML_CACHE_VERIFY;
      else
         mode = ML_CACHE_ON;
   }
   return (enum ml_cache_mode)mode;
}

static const char *
ml_cache_dir(void)
{
   static char dir[PATH_MAX];

   if (ml_cache_mode() == ML_CACHE_OFF || (!dir[0] && !npu_cache_default_dir(dir, sizeof(dir))))
      return NULL;
   return dir;
}

static void
ml_cache_digest(const void *data, size_t size, uint8_t digest[NPU_CACHE_DIGEST_SIZE])
{
   _mesa_blake3_compute(data, size, digest);
}

/* Bump when the key or the stored blob changes in a way the hashed data doesn't show. */
#define ML_CACHE_KEY_SCHEME "alta-ml-cache-key-2 " PACKAGE_VERSION

/*
 * The source text of the code that compiles weights (see etnaviv_ml_nn.h). Its
 * hash stands in the key where the library's build ID used to, so only a change
 * to these files, or a new Mesa version, makes old entries unreachable.
 */
static const struct {
   const char *name;
   const uint8_t *start, *end;
} ml_cache_sources[] = {
   { "etnaviv_ml_nn_v7.c", etna_ml_src_nn_v7_c, etna_ml_src_nn_v7_c_end },
   { "etnaviv_ml_nn_v8.c", etna_ml_src_nn_v8_c, etna_ml_src_nn_v8_c_end },
   { "etnaviv_ml_nn.c", etna_ml_src_nn_c, etna_ml_src_nn_c_end },
   { "etnaviv_ml.h", etna_ml_src_ml_h, etna_ml_src_ml_h_end },
   { "etnaviv_ml_nn.h", etna_ml_src_nn_h, etna_ml_src_nn_h_end },
   { "etna_core_info.h", etna_ml_src_core_info_h, etna_ml_src_core_info_h_end },
};

static void
ml_cache_compiler_fingerprint(blake3_hash digest)
{
   blake3_hasher ctx;

   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, ML_CACHE_KEY_SCHEME, sizeof(ML_CACHE_KEY_SCHEME));
   for (unsigned i = 0; i < ARRAY_SIZE(ml_cache_sources); i++) {
      uint64_t size = ml_cache_sources[i].end - ml_cache_sources[i].start;

      _mesa_blake3_update(&ctx, ml_cache_sources[i].name, strlen(ml_cache_sources[i].name) + 1);
      _mesa_blake3_update(&ctx, &size, sizeof(size));
      _mesa_blake3_update(&ctx, ml_cache_sources[i].start, size);
   }
   _mesa_blake3_final(&ctx, digest);
}

static bool
ml_cache_key(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
             char hex[NPU_CACHE_KEY_CHARS])
{
   blake3_hash compiler;
   struct etna_screen *screen = subgraph->screen;
   struct etna_operation scrubbed;
   blake3_hasher ctx;
   blake3_hash digest;

   ml_cache_compiler_fingerprint(compiler); /* about 100 KiB of text: well under a millisecond */

   /* Hash the whole operation so a field can't be forgotten, minus its pointers, which change every run. */
   memcpy(&scrubbed, operation, sizeof(scrubbed));
   memset(&scrubbed.link, 0, sizeof(scrubbed.link));
   scrubbed.weight_tensor = NULL;
   scrubbed.bias_tensor = NULL;

   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, compiler, sizeof(compiler));
   _mesa_blake3_update(&ctx, screen->info, sizeof(*screen->info));
   _mesa_blake3_update(&ctx, etna_ml_get_core_info(screen), sizeof(struct etna_core_npu_info));
   _mesa_blake3_update(&ctx, &screen->specs.nn_core_version, sizeof(screen->specs.nn_core_version));
   _mesa_blake3_update(&ctx, &scrubbed, sizeof(scrubbed));
   if (operation->weight_tensor)
      _mesa_blake3_update(&ctx, operation->weight_tensor, operation->weight_tensor_size);
   if (operation->bias_tensor)
      _mesa_blake3_update(&ctx, operation->bias_tensor, operation->bias_tensor_size);
   _mesa_blake3_final(&ctx, digest);

   for (unsigned i = 0; i < NPU_CACHE_DIGEST_SIZE; i++)
      snprintf(hex + i * 2, 3, "%02x", digest[i]);

   return true;
}

struct etna_bo *
etna_ml_disk_cache_load(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
                        unsigned *cache_size)
{
   const char *dir = ml_cache_dir();
   char hex[NPU_CACHE_KEY_CHARS];
   struct ml_cache_blob_header header;
   size_t size = 0;

   if (!dir || !ml_cache_key(subgraph, operation, hex))
      return NULL;

   uint8_t *payload = npu_cache_read(dir, hex, ml_cache_digest, &size);

   if (payload && size >= sizeof(header))
      memcpy(&header, payload, sizeof(header));
   if (!payload || size < sizeof(header) || size != sizeof(header) + header.data_size ||
       header.data_size > header.bo_size) {
      free(payload);
      ml_cache_counts.misses++;
      return NULL;
   }

   struct etna_bo *bo = etna_ml_create_bo(subgraph->screen, header.bo_size); /* zeroed */

   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_WRITE);
   memcpy(etna_bo_map(bo), payload + sizeof(header), header.data_size);
   etna_bo_cpu_fini(bo);
   free(payload);

   *cache_size = header.cache_size;
   ml_cache_counts.hits++;

   return bo;
}

void
etna_ml_disk_cache_store(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
                         struct etna_bo *bo, unsigned cache_size)
{
   const char *dir = ml_cache_dir();
   char hex[NPU_CACHE_KEY_CHARS];
   struct ml_cache_blob_header header = { .bo_size = etna_bo_size(bo), .cache_size = cache_size };

   if (!dir || !ml_cache_key(subgraph, operation, hex) || !npu_cache_mkdirs(dir))
      return;

   uint8_t *payload = malloc(sizeof(header) + header.bo_size);
   if (!payload)
      return;

   uint8_t *data = payload + sizeof(header);

   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_READ);
   memcpy(data, etna_bo_map(bo), header.bo_size);
   etna_bo_cpu_fini(bo);

   /* The buffer starts zeroed, so its zero tail needn't be stored. */
   header.data_size = header.bo_size;
   while (header.data_size > 0 && data[header.data_size - 1] == 0)
      header.data_size--;
   memcpy(payload, &header, sizeof(header));

   size_t payload_size = sizeof(header) + header.data_size;

   if (npu_cache_trim(dir, payload_size + NPU_CACHE_FILE_OVERHEAD, NULL) &&
       npu_cache_write(dir, hex, ml_cache_digest, payload, payload_size))
      ml_cache_counts.stored++;

   free(payload);
}

bool
etna_ml_disk_cache_verifying(void)
{
   return ml_cache_mode() == ML_CACHE_VERIFY;
}

void
etna_ml_disk_cache_check(struct etna_bo *cached, unsigned cached_cache_size,
                         struct etna_bo *fresh, unsigned fresh_cache_size)
{
   uint32_t size = etna_bo_size(fresh);
   bool same;

   etna_bo_cpu_prep(cached, DRM_ETNA_PREP_READ);
   etna_bo_cpu_prep(fresh, DRM_ETNA_PREP_READ);
   same = etna_bo_size(cached) == size && cached_cache_size == fresh_cache_size &&
          memcmp(etna_bo_map(cached), etna_bo_map(fresh), size) == 0;
   etna_bo_cpu_fini(fresh);
   etna_bo_cpu_fini(cached);

   if (!same) {
      fprintf(stderr, "etnaviv/ml: disk cache verify FAILED: a cached entry differs from a fresh compile\n");
      abort();
   }
}

/* Shrinks the cache if the disk has filled up since the last program ran. */
static void
ml_cache_trim(void)
{
   const char *dir = ml_cache_dir();
   struct npu_cache_stats stats = {0};

   static bool warned;

   if (dir && npu_cache_trim(dir, 0, &stats) && stats.low_space && !warned) {
      warned = true; /* once per program, not once per subgraph */
      fprintf(stderr, "etnaviv/ml: WARNING: disk space is low: %.1f GiB free, under 10 GiB. The NPU weight cache "
                      "(%.1f MiB) won't grow until space is freed%s\n",
              stats.avail / 1073741824.0, stats.used / 1048576.0,
              stats.evicted ? "; older entries were evicted" : "");
   }
}

static void
ml_cache_report(void)
{
   const char *dir = ml_cache_dir();
   struct npu_cache_stats stats = {0};

   if (dir && ml_cache_mode() >= ML_CACHE_VERBOSE && npu_cache_trim(dir, 0, &stats)) {
      blake3_hash compiler;

      ml_cache_compiler_fingerprint(compiler);
      fprintf(stderr, "etnaviv/ml: disk cache: %u hits, %u misses, %u stored; holding %.1f MiB, limit %.0f MiB; "
                      "weight compiler %02x%02x%02x%02x\n",
              ml_cache_counts.hits, ml_cache_counts.misses, ml_cache_counts.stored,
              stats.used / 1048576.0, stats.budget / 1048576.0,
              compiler[0], compiler[1], compiler[2], compiler[3]);
   }
}

/* alta: flag from the matching etnaviv kernel build, not upstream UAPI. */
#define DRM_ETNA_GEM_ALTA_AXI_SRAM 0x00200000
#define AXI_SRAM_SIZE (1024 * 1024)
#define AXI_SRAM_PAGE 4096

static struct pipe_resource *
create_axi_sram_resource(struct etna_ml_subgraph *subgraph)
{
   struct etna_screen *screen = subgraph->screen;
   struct etna_bo *bo = etna_bo_new(screen->dev, AXI_SRAM_SIZE,
                                    DRM_ETNA_GEM_CACHE_WC | DRM_ETNA_GEM_ALTA_AXI_SRAM);
   if (!bo)
      return NULL;

   struct pipe_resource *res = pipe_buffer_create(&screen->base, 0, PIPE_USAGE_DEFAULT, AXI_SRAM_SIZE);
   struct etna_buffer_resource *buffer = etna_buffer_resource(res);
   etna_bo_del(buffer->bo);
   buffer->bo = bo;

   return res;
}

/*
 * alta: opt-in with ETNA_ML_AXI_SRAM=<max tensors> or =all. Moves intermediate
 * tensors into the NPU's AXI SRAM, so they stop going through DDR. Graph inputs
 * and outputs stay in RAM because the CPU touches them, and aliased tensors stay
 * because they share a buffer. Each invoke is one submit, so tensors can't be
 * clobbered between operations.
 */
static void
place_tensors_in_axi_sram(struct etna_ml_subgraph *subgraph, struct list_head *operations)
{
   const char *env = getenv("ETNA_ML_AXI_SRAM");
   unsigned max_placed = !env ? 0 : (!strcmp(env, "all") ? UINT_MAX : (unsigned)atoi(env));

   if (!max_placed)
      return;

   unsigned tensor_count = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);
   int *writer = malloc(tensor_count * sizeof(*writer));
   int *last_reader = malloc(tensor_count * sizeof(*last_reader));
   bool *excluded = calloc(tensor_count, sizeof(*excluded));
   struct axi_sram_item *items = calloc(tensor_count, sizeof(*items));
   unsigned *item_tensor = calloc(tensor_count, sizeof(*item_tensor));
   unsigned n = 0, op = 0;

   for (unsigned t = 0; t < tensor_count; t++)
      writer[t] = last_reader[t] = -1;

   list_for_each_entry(struct etna_operation, operation, operations, link) {
      bool plain = operation->type == ETNA_JOB_TYPE_NN || operation->type == ETNA_JOB_TYPE_TP;

      /* Some lowerings leave a count at 0 when there is one tensor. */
      for (unsigned i = 0; i < MAX2(operation->input_count, 1); i++) {
         unsigned t = operation->input_tensors[i];
         if (plain)
            last_reader[t] = op;
         else
            excluded[t] = true;
      }
      for (unsigned i = 0; i < MAX2(operation->output_count, 1); i++) {
         unsigned t = operation->output_tensors[i];
         if (!plain || writer[t] >= 0)
            excluded[t] = true;
         else
            writer[t] = op;
      }
      op++;
   }

   for (unsigned t = 0; t < tensor_count; t++) {
      bool shared = false;

      if (!tensors[t] || !tensors[t]->resource || excluded[t] || writer[t] < 0 ||
          last_reader[t] <= writer[t])
         continue;
      for (unsigned u = 0; u < tensor_count && !shared; u++)
         shared = u != t && tensors[u] && tensors[u]->resource == tensors[t]->resource;
      if (shared)
         continue;

      unsigned k = n++;
      while (k > 0 && items[k - 1].first_use > (unsigned)writer[t]) {
         items[k] = items[k - 1];
         item_tensor[k] = item_tensor[k - 1];
         k--;
      }
      items[k] = (struct axi_sram_item){ .size = align(tensors[t]->size, AXI_SRAM_PAGE),
                                         .first_use = writer[t], .last_use = last_reader[t] };
      item_tensor[k] = t;
   }

   unsigned placed = axi_sram_assign(items, n, AXI_SRAM_SIZE, max_placed);
   struct pipe_resource *sram = placed ? create_axi_sram_resource(subgraph) : NULL;

   if (sram) {
      unsigned top = 0;

      for (unsigned k = 0; k < n; k++) {
         if (items[k].offset < 0)
            continue;
         struct etna_ml_tensor *tensor = tensors[item_tensor[k]];
         pipe_resource_reference(&tensor->resource, sram);
         tensor->offset = items[k].offset;
         top = MAX2(top, items[k].offset + items[k].size);
      }
      subgraph->axi_sram = sram;
      pipe_resource_reference(&sram, NULL); /* the tensors hold references now */
      fprintf(stderr, "etnaviv/ml: AXI SRAM: placed %u of %u intermediate tensors, using %u of %u KiB\n",
              placed, n, top / 1024, AXI_SRAM_SIZE / 1024);
   } else if (placed) {
      fprintf(stderr, "etnaviv/ml: AXI SRAM not available (is /sys/module/etnaviv/parameters/axi_sram 1?), using RAM\n");
   }

   free(writer);
   free(last_reader);
   free(excluded);
   free(items);
   free(item_tensor);
}

struct pipe_ml_subgraph *
etna_ml_subgraph_create(struct pipe_ml_device *pdevice,
                        const struct pipe_ml_operation *poperations,
                        unsigned count)
{
   struct etna_screen *screen = etna_ml_device_screen(pdevice);
   unsigned nn_core_count = etna_ml_get_core_info(screen)->nn_core_count;
   struct etna_ml_subgraph *subgraph;
   struct list_head operations;
   unsigned tensor_count;

   if (nn_core_count < 1) {
      fprintf(stderr, "We need at least 1 NN core to do anything useful.\n");
      abort();
   }

   subgraph = calloc(1, sizeof(*subgraph));
   tensor_count = count_tensors(poperations, count);

   list_inithead(&operations);

   subgraph->base.device = pdevice;
   subgraph->screen = screen;
   subgraph->operations = UTIL_DYNARRAY_INIT;

   subgraph->tensors = UTIL_DYNARRAY_INIT;
   if (!util_dynarray_resize(&subgraph->tensors, struct etna_ml_tensor*, tensor_count))
      return NULL;
   memset(util_dynarray_begin(&subgraph->tensors), 0, subgraph->tensors.size);

   ml_cache_trim();
   lower_operations(subgraph, poperations, count, &operations);
   place_tensors_in_axi_sram(subgraph, &operations);

   list_for_each_entry(struct etna_operation, operation, &operations, link) {
      struct etna_vip_instruction instruction = {0};

      switch(operation->type) {
         case ETNA_JOB_TYPE_NN:
            etna_ml_compile_operation_nn(subgraph, operation, &instruction);
            break;
         case ETNA_JOB_TYPE_TP:
            etna_ml_compile_operation_tp(subgraph, operation, &instruction);
            break;
         case ETNA_JOB_TYPE_SHADER:
            if (etna_ml_compile_operation_shader(subgraph, operation, &instruction))
               break;
            if (operation->shader_op == ETNA_ML_SHADER_TRANSPOSE) {
               /* lowered as a TP transpose/detranspose, so the TP can take it if the shader core can't */
               etna_ml_compile_operation_tp(subgraph, operation, &instruction);
               break;
            }
            /* the support check accepted it, so this shouldn't happen; the output stays unwritten */
            BUG("ML shader operation could not be compiled; its output will be wrong");
            continue;
         case ETNA_JOB_TYPE_CONCAT:
         case ETNA_JOB_TYPE_SPLIT:
         case ETNA_JOB_TYPE_BYPASS:
            continue;
      }

      util_dynarray_append(&subgraph->operations, instruction);
   }

   ml_cache_report();

   list_for_each_entry_safe(struct etna_operation, operation, &operations, link) {
      free(operation->weight_tensor);
      free(operation->bias_tensor);
      free(operation);
   }

   return &subgraph->base;
}

static void
dump_buffer(const uint8_t *ptr, char *name, int operation_nr, int suboperation_nr, int offset, unsigned size)
{
   char buffer[255];

   snprintf(buffer, sizeof(buffer), "mesa-%s-%03u-%03u.bin", name, operation_nr, suboperation_nr);

   ML_DBG("Dumping buffer from 0x%lx at offset %d with size %d to %s\n", ptr, offset, size, buffer);

   FILE *f = fopen(buffer, "wb");
   assert(f);
   fwrite(ptr + offset, 1, size, f);
   if(ferror(f)) {
      ML_DBG("Error in writing to file: %s\n", strerror(errno));
   }
   fflush(f);
   fclose(f);
}

static void
dump_bo(struct etna_bo *bo, char *name, int operation_nr, int suboperation_nr, int offset, int size)
{
   const uint8_t *map = etna_bo_map(bo);
   if (size == 0)
      size = etna_bo_size(bo) - offset;
   dump_buffer(map, name, operation_nr, suboperation_nr, offset, size);
}

static void
init_npu(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_cmd_stream *stream = ctx->stream;

   /* These zeroes match the blob's cmdstream. They are here to make diff'ing easier.*/
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);

   etna_set_state(stream, VIVS_PA_SYSTEM_MODE, VIVS_PA_SYSTEM_MODE_PROVOKING_VERTEX_LAST |
                                               VIVS_PA_SYSTEM_MODE_HALF_PIXEL_CENTER);
   etna_set_state(stream, VIVS_GL_API_MODE, VIVS_GL_API_MODE_OPENCL);

   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);

   pctx->flush(pctx, NULL, 0);
}

static void
close_batch(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_cmd_stream *stream = ctx->stream;

   unsigned cache = VIVS_GL_FLUSH_CACHE_DEPTH | VIVS_GL_FLUSH_CACHE_COLOR | VIVS_GL_FLUSH_CACHE_UNK10;
   if (!DBG_ENABLED(ETNA_DBG_NPU_PARALLEL))
      cache |= VIVS_GL_FLUSH_CACHE_UNK11 | VIVS_GL_FLUSH_CACHE_SHADER_L1;

   etna_set_state(stream, VIVS_GL_FLUSH_CACHE, cache);
   etna_set_state(stream, VIVS_GL_FLUSH_CACHE, cache);

   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);

   ctx->dirty = 0;
}

void
etna_ml_subgraph_invoke(struct pipe_context *pctx, struct pipe_ml_subgraph *psubgraph,
                        unsigned inputs_count, unsigned input_idxs[], void *inputs[],
                        bool is_signed[])
{
   struct etna_context *ctx = etna_context(pctx);
   unsigned tp_core_count = etna_ml_get_core_info(ctx->screen)->tp_core_count;
   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)(psubgraph);
   struct etna_cmd_stream *stream = ctx->stream;
   static bool is_initialized = false;

   if (!is_initialized) {
      init_npu(pctx);
      is_initialized = true;
   }

   if (!DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING)) {
      /* These zeroes match the blob's cmdstream. They are here to make diff'ing easier.*/
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
   }

   for (int i = 0; i < inputs_count; i++) {
      struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, input_idxs[i]);
      if (is_signed[i]) {
         struct pipe_transfer *dst_transfer;
         const uint8_t *src = inputs[i];
         uint8_t *dst_map;
         dst_map = pipe_buffer_map_range(pctx, tensor->resource, tensor->offset, tensor->size, PIPE_MAP_WRITE, &dst_transfer);
         assert(dst_map);
         for (unsigned k = 0; k < tensor->size; k++) {
            dst_map[k] = src[k] + 128;
         }
         pipe_buffer_unmap(pctx, dst_transfer);
      } else {
         pipe_buffer_write(pctx, tensor->resource, tensor->offset, tensor->size, inputs[i]);
      }
   }

   static unsigned i = 0;
   util_dynarray_foreach(&subgraph->operations, struct etna_vip_instruction, operation) {

      if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS)) {
         struct pipe_transfer *transfer = NULL;
         pipe_buffer_map(pctx, operation->input, PIPE_MAP_READ, &transfer);
         dump_bo(etna_buffer_resource(operation->input)->bo, "input", i, 0, operation->input_offset, 0);
         pipe_buffer_unmap(pctx, transfer);

         switch (operation->type) {
            case ETNA_JOB_TYPE_TP:
               for (unsigned j = 0; j < tp_core_count && operation->configs[j]; j++) {
                  dump_bo(operation->configs[j], "tp", i, j, 0, 0);
                  if (operation->pwl_lut)
                     dump_bo(operation->pwl_lut, "lut", i, j, 0, 0);
               }
               break;
            case ETNA_JOB_TYPE_NN:
               dump_bo(operation->configs[0], "nn", i, 0, 0, 0);
               dump_bo(operation->coefficients, "compressed", i, 0, 0, 0);
               break;
            case ETNA_JOB_TYPE_SHADER:
               break; /* the shader and command stream are dumped by the compute code */
            default:
               UNREACHABLE("Unsupported ML operation type");
         }
      }

      if (DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING)) {
         /* These zeroes match the blob's cmdstream. They are here to make diff'ing easier.*/
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
      }

      for (unsigned j = 0; j < tp_core_count && operation->configs[j]; j++)
         etna_cmd_stream_ref_bo(stream, operation->configs[j], ETNA_RELOC_READ);
      if (operation->coefficients)
         etna_cmd_stream_ref_bo(stream, operation->coefficients, ETNA_RELOC_READ);
      if (operation->pwl_lut)
         etna_cmd_stream_ref_bo(stream, operation->pwl_lut, ETNA_RELOC_READ);
      etna_cmd_stream_ref_bo(stream, etna_buffer_resource(operation->input)->bo, ETNA_RELOC_READ);
      etna_cmd_stream_ref_bo(stream, etna_buffer_resource(operation->output)->bo, ETNA_RELOC_WRITE);

      switch (operation->type) {
         case ETNA_JOB_TYPE_TP:
            etna_ml_emit_operation_tp(pctx, subgraph, operation, i);
            break;
         case ETNA_JOB_TYPE_NN:
            etna_ml_emit_operation_nn(pctx, subgraph, operation, i);
            break;
         case ETNA_JOB_TYPE_SHADER:
            etna_ml_emit_operation_shader(pctx, subgraph, operation, i);
            break;
         default:
            UNREACHABLE("Unsupported ML operation type");
      }

      if (DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING)) {
         ML_DBG("Running operation %d - %d\n", i, operation->type);
         close_batch(pctx);

         if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS))
            dump_buffer((uint8_t *)ctx->stream->buffer, "cmd", i, 0, 0, ctx->stream->offset * 4);

         pctx->flush(pctx, NULL, 0);

         if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS)) {
            struct pipe_transfer *transfer = NULL;

            pipe_buffer_map(pctx, operation->output, PIPE_MAP_READ, &transfer);
            dump_bo(etna_buffer_resource(operation->output)->bo, "output", i, 0, operation->output_offset, 0);
            pipe_buffer_unmap(pctx, transfer);
         }

         stream = ctx->stream;
      }

      i++;
   }

   if (!DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING))
      close_batch(pctx);

   if (DBG_ENABLED(ETNA_DBG_FLUSH_ALL))
      pctx->flush(pctx, NULL, 0);
}

void
etna_ml_subgraph_read_outputs(struct pipe_context *context, struct pipe_ml_subgraph *psubgraph,
                              unsigned outputs_count, unsigned output_idxs[], void *outputs[],
                              bool is_signed[])
{
   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)(psubgraph);
   unsigned operation_count = util_dynarray_num_elements(&subgraph->operations, struct etna_vip_instruction);
   struct etna_vip_instruction *last_operation;

   if (operation_count > 0) {
      last_operation = util_dynarray_element(&subgraph->operations,
                                             struct etna_vip_instruction,
                                             operation_count - 1);

      if (DBG_ENABLED(ETNA_DBG_ML_MSGS)) {
         long start, end;
         struct timespec time;

         clock_gettime(CLOCK_MONOTONIC, &time);
         start = (long)time.tv_sec * 1000 + (long)time.tv_nsec / 1000000;

         context->flush(context, NULL, 0);

         struct pipe_transfer *transfer = NULL;
         pipe_buffer_map(context, last_operation->output, PIPE_MAP_READ, &transfer);
         pipe_buffer_unmap(context, transfer);

         clock_gettime(CLOCK_MONOTONIC, &time);
         end = (long)time.tv_sec * 1000 + (long)time.tv_nsec / 1000000;
         ML_DBG("Running the NN job took %ld ms.\n", (end - start));
      } else
         context->flush(context, NULL, 0);
   }

   for (int i = 0; i < outputs_count; i++) {
      struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);
      if (res == subgraph->axi_sram) {
         fprintf(stderr, "etnaviv/ml: output tensor %u is in the AXI SRAM, which the CPU can't read. Unset ETNA_ML_AXI_SRAM.\n",
                 output_idxs[i]);
         abort();
      }
      if (is_signed[i]) {
         struct pipe_transfer *src_transfer;
         uint8_t *src_map;
         src_map = (uint8_t *) pipe_buffer_map_range(context,
                                                     res,
                                                     0, pipe_buffer_size(res),
                                                     PIPE_MAP_READ,
                                                     &src_transfer);
         assert(src_map);
         for (unsigned k = 0; k < etna_ml_get_size(subgraph, output_idxs[i]); k++) {
            ((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;
         }
         pipe_buffer_unmap(context, src_transfer);
      } else {
         pipe_buffer_read(context, res, 0, etna_ml_get_size(subgraph, output_idxs[i]), outputs[i]);
      }
   }
}

void
etna_ml_subgraph_destroy(struct pipe_ml_device *pdevice, struct pipe_ml_subgraph *psubgraph)
{
   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)(psubgraph);

   util_dynarray_foreach(&subgraph->operations, struct etna_vip_instruction, operation) {
      for (unsigned j = 0; j < MAX_CONFIG_BOS && operation->configs[j]; j++)
         etna_bo_del(operation->configs[j]);
      etna_bo_del(operation->coefficients);
      etna_bo_del(operation->pwl_lut);
      pipe_resource_reference(&operation->input, NULL);
      pipe_resource_reference(&operation->output, NULL);
      etna_ml_shader_instruction_fini(operation);
   }
   util_dynarray_fini(&subgraph->operations);

   util_dynarray_foreach(&subgraph->tensors, struct etna_ml_tensor*, tensor) {
      if (!*tensor)
         continue;
      pipe_resource_reference(&(*tensor)->resource, NULL);
      free(*tensor);
   }
   util_dynarray_fini(&subgraph->tensors);

   free(subgraph);
}
