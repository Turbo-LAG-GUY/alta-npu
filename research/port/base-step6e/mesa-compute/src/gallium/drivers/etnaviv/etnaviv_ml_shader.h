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

/* MEAN on the shader core (ETNA_ML_SHADER=mean) */
bool
etna_ml_shader_mean_supported(struct etna_screen *screen, const struct pipe_ml_operation *poperation);

void
etna_ml_lower_mean(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
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

/* MEAN's requantization, exact to XNNPACK (the CPU path beside Teflon): for S = the sum of the n stored bytes of one
 * output (int8 is stored +128), the stored output byte is q0 + (S >= table[q0 + 1]) - (S < table[q0]) with the
 * estimate q0 = clamp((S * m + b) >> sh, 0, 255); table[k] = the smallest S whose output is >= k, table[0] = 0,
 * table[256] = UINT32_MAX. etna_ml_mean_requant checks every S in 0..255n (false: no exact form, stay on the CPU). */
struct etna_ml_mean_requant {
   int32_t m, b;
   unsigned sh;
   uint32_t table[257];
};

bool
etna_ml_mean_requant(unsigned n, float in_scale, int in_zp, float out_scale, int out_zp, bool is_signed,
                     struct etna_ml_mean_requant *rq);

/* XNNPACK's quantized mean of n values whose sum (in the tensors' own domain) is s */
int
etna_ml_mean_xnnpack(int64_t s, unsigned n, float in_scale, int in_zp, float out_scale, int out_zp, bool is_signed);

/* MEAN over the channels of NCHW planes: positions = H*W (% 4 == 0) planes' length, one thread per 4 positions,
 * a loop over the channels. Kernel input: out, in, table, channels, plane bytes (= positions). */
struct nir_shader *
etna_ml_shader_build_mean_channels(const struct nir_shader_compiler_options *opts, const struct etna_ml_mean_requant *rq);

/* MEAN over the spatial dimensions: the average pool kernel with the requantization table instead of its
 * division. Kernel input: out, in, table. */
struct nir_shader *
etna_ml_shader_build_mean_spatial(const struct nir_shader_compiler_options *opts, unsigned hw, unsigned channels,
                                  const struct etna_ml_mean_requant *rq);

#endif
