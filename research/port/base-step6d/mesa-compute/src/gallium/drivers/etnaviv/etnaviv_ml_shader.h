/*
 * SPDX-License-Identifier: MIT
 */

#ifndef H_ETNA_ML_SHADER
#define H_ETNA_ML_SHADER

#include "etnaviv_ml.h"

struct etna_screen;

/* operation: an implicit TP transpose/detranspose; true if it should run on the shader core instead */
bool
etna_ml_shader_wants_transpose(const struct etna_ml_subgraph *subgraph, const struct etna_operation *operation);

/* false: the operation can't run on the shader core after all (alignment, AXI SRAM, compile); compile it for the TP */
bool
etna_ml_compile_operation_shader(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
                                 struct etna_vip_instruction *instruction);

void
etna_ml_emit_operation_shader(struct pipe_context *pctx, struct etna_ml_subgraph *subgraph,
                              struct etna_vip_instruction *operation, unsigned idx);

void
etna_ml_shader_screen_fini(struct etna_screen *screen);

/* softmax on the shader core (ETNA_ML_SHADER=softmax) */
bool
etna_ml_shader_softmax_enabled(const struct etna_screen *screen);

bool
etna_ml_shader_softmax_supported(struct etna_screen *screen, const struct pipe_ml_operation *poperation);

void
etna_ml_lower_softmax(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
                      struct etna_operation *operation);

/* global average pool on the shader core (ETNA_ML_SHADER=avgpool) */
bool
etna_ml_shader_avgpool_supported(struct etna_screen *screen, const struct pipe_ml_operation *poperation);

void
etna_ml_lower_avgpool(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
                      struct etna_operation *operation);

void
etna_ml_shader_instruction_fini(struct etna_vip_instruction *instruction);

/* softmax kernel (see etnaviv_ml_shader.c): rows (grid size) of n uint8, factor = beta * input_scale * log2(e) */
#define ETNA_ML_SOFTMAX_THREADS 64 /* the most threads per workgroup tried */

struct nir_shader;
struct nir_shader_compiler_options;

struct nir_shader *
etna_ml_shader_build_softmax(const struct nir_shader_compiler_options *opts, unsigned n, unsigned rows,
                             float factor, unsigned threads);

/* global average pool kernel: channel planes of hw uint8 (NCHW), channels % 4 == 0, one thread per 4 channels
 * (grid size channels / 4), out[c] = (sum of plane c + hw / 2) / hw. NULL if hw is outside 1..ETNA_ML_AVGPOOL_MAX_HW. */
#define ETNA_ML_AVGPOOL_MAX_HW 256

struct nir_shader *
etna_ml_shader_build_avgpool(const struct nir_shader_compiler_options *opts, unsigned hw, unsigned channels);

#endif
