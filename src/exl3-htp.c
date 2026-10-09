#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "hex-utils.h"
#include "hvx-base.h"
#include "hvx-copy.h"
#include "matmul-ops.h"
#include "hmx-utils.h"
#include "exl3-block.h"
#define EXL3_HVX_HMX_OUTPUT
#include "hexagon/exl3-hvx.h"
#include <math.h>
#include <string.h>

static float half_value(const uint16_t * ptr) {
    _Float16 h;
    memcpy(&h, ptr, 2);
    return (float) h;
}

static void had128(float * x) {
    for (unsigned step = 1; step < 128; step *= 2) {
        for (unsigned base = 0; base < 128; base += 2 * step) for (unsigned j = 0; j < step; ++j) {
            const float a = x[base + j], b = x[base + j + step];
            x[base + j] = a + b; x[base + j + step] = a - b;
        }
    }
    for (unsigned j = 0; j < 128; ++j) x[j] *= 0.08838834764831844055f;
}

static uint64_t offset_row(const struct htp_tensor * tensor, unsigned row) {
    const unsigned i1 = row % tensor->ne[1];
    row /= tensor->ne[1];
    const unsigned i2 = row % tensor->ne[2], i3 = row / tensor->ne[2];
    return (uint64_t) i1 * tensor->nb[1] + (uint64_t) i2 * tensor->nb[2] + (uint64_t) i3 * tensor->nb[3];
}

struct exl3_task {
    struct htp_ops_context * octx;
    unsigned bits, k, n, rows, first_row;
    size_t per_thread;
    float * xh;
    unsigned char * scratch;
    bool hmx;
    uint16_t * act_half;
    uint16_t * scales;
    qurt_mutex_t * hmx_lock;
};

struct exl3_hmx_job { uint16_t * output; const uint16_t * input; const uint16_t * weight; const uint16_t * scales; };
static void exl3_hmx_compute(void * opaque) {
    const struct exl3_hmx_job * job = opaque;
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned) job->scales));
    for (unsigned col = 0; col < 4; ++col) {
        const uint16_t * weight = job->weight + col * 4096;
        uint16_t * output = job->output + col * 1024;
        asm volatile(HMX_CLRACC_F16());
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") :: "r"(8191), "r"(job->input), "r"(weight));
        asm volatile(HMX_STORE_AFTER_F16("%0", "%1") :: "r"(output), "r"(0) : "memory");
    }
}

static void exl3_worker(unsigned nth, unsigned ith, void * opaque) {
    struct exl3_task * task = (struct exl3_task *) opaque;
    const struct htp_tensor * weight = task->octx->src[0];
    const struct htp_tensor * output = task->octx->dst;
    struct htp_thread_trace * trace = &task->octx->ctx->trace[ith];
    unsigned char * local = task->scratch + ith * task->per_thread;
    uint16_t * decoded = (uint16_t *) local;
    memset(decoded + 128 * 128, 0, EXL3_DECODE_PAD_BYTES);
    float * result = (float *) (decoded + 128 * 128 + EXL3_DECODE_PAD_BYTES / 2);
    uint16_t * packed = (uint16_t *) (result + task->rows * 128);
    HVX_Vector * gathered = (HVX_Vector *) ((unsigned char *) packed + exl3_group_bytes(task->bits));
    uint16_t * hmx_output = (uint16_t *) (local + exl3_hmx_thread_bytes(task->bits, task->rows) - 8192u);
    for (unsigned nb = ith; nb < task->n / 128; nb += nth) {
        memset(result, 0, task->rows * 128 * 4);
        for (unsigned kb = 0; kb < task->k / 128; ++kb) {
            htp_trace_event_start(trace, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) kb);
            const unsigned char * group = (const unsigned char *) (uintptr_t) weight->data +
                ((size_t) nb * (task->k / 128) + kb) * exl3_group_bytes(task->bits);
            memcpy(packed, group, 2048u * task->bits);
            for (unsigned r = 0; r < 8; ++r) for (unsigned c = 0; c < 8; ++c) {
                exl3_hvx_decode_tile(packed + (r * 8 + c) * 16 * task->bits, task->bits, NULL, gathered,
                                     decoded + ((c / 2) * 4 + r / 2) * 1024 + (r % 2) * 512 + (c % 2) * 32);
            }
            htp_trace_event_stop(trace, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) kb);
            htp_trace_event_start(trace, HTP_TRACE_EVT_HVX_COMP, (uint16_t) kb);
            if (task->hmx) {
                struct exl3_hmx_job job = {hmx_output, task->act_half + kb * 4096, decoded, task->scales};
                qurt_mutex_lock(task->hmx_lock);
                if (!hmx_queue_push(task->octx->ctx->hmx_queue, hmx_queue_make_desc(exl3_hmx_compute, &job))) {
                    htp_ops_context_set_status(task->octx, HTP_STATUS_INTERNAL_ERR);
                    qurt_mutex_unlock(task->hmx_lock);
                    return;
                }
                hmx_queue_pop(task->octx->ctx->hmx_queue);
                qurt_mutex_unlock(task->hmx_lock);
                for (unsigned row = 0; row < task->rows; ++row) for (unsigned col = 0; col < 4; ++col) {
                    const HVX_VectorPair pair = hvx_vec_f16_to_f32_shuff(hvx_vmem(hmx_output + col * 1024 + (row / 2) * 64));
                    const HVX_Vector value = row % 2 ? Q6_V_hi_W(pair) : Q6_V_lo_W(pair);
                    HVX_Vector * dst = (HVX_Vector *) (result + row * 128);
                    dst[col] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(dst[col], value));
                }
            } else {
                for (unsigned i = 0; i < 128; ++i) {
                    HVX_Vector w[4];
                    for (unsigned col = 0; col < 4; ++col) {
                        const uint16_t * strip = decoded + (col * 4 + i / 32) * 1024 + ((i % 32) / 2) * 64;
                        const HVX_VectorPair pair = hvx_vec_f16_to_f32_shuff(hvx_vmem(strip));
                        w[col] = i % 2 ? Q6_V_hi_W(pair) : Q6_V_lo_W(pair);
                    }
                    for (unsigned row = 0; row < task->rows; ++row) {
                        uint32_t scalar;
                        memcpy(&scalar, task->xh + row * task->k + kb * 128 + i, 4);
                        HVX_Vector x = Q6_V_vsplat_R(scalar);
                        HVX_Vector * z = (HVX_Vector *) (result + row * 128);
                        for (unsigned v = 0; v < 4; ++v) {
                            z[v] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(w[v], x), z[v]));
                        }
                    }
                }
            }
            htp_trace_event_stop(trace, HTP_TRACE_EVT_HVX_COMP, (uint16_t) kb);
        }
        const unsigned char * group = (const unsigned char *) (uintptr_t) weight->data +
            (size_t) nb * (task->k / 128) * exl3_group_bytes(task->bits);
        const uint16_t * sv = exl3_group_sv(group, task->bits);
        for (unsigned row = 0; row < task->rows; ++row) {
            had128(result + row * 128);
            float * dst = (float *) ((uintptr_t) output->data + offset_row(output, task->first_row + row)) + nb * 128;
            for (unsigned j = 0; j < 128; ++j) dst[j] = result[row * 128 + j] * half_value(sv + j);
        }
    }
}

int op_exl3_matmul(struct htp_ops_context * octx) {
    if (octx->op != HTP_OP_MUL_MAT) return HTP_STATUS_INVAL_PARAMS;
    const struct htp_tensor * weight = octx->src[0], * input = octx->src[1], * output = octx->dst;
    if (!weight || !input || !output) return HTP_STATUS_INVAL_PARAMS;
    const unsigned bits = exl3_type_bits(weight->type), k = weight->ne[0], n = weight->ne[1];
    uint64_t rows64 = 1;
    for (unsigned d = 1; d < 4; ++d) {
        if (!input->ne[d] || input->ne[d] > 1024 || rows64 > 1024u / input->ne[d] || output->ne[d] != input->ne[d]) return HTP_STATUS_INVAL_PARAMS;
        rows64 *= input->ne[d];
    }
    if (!bits || !k || !n || k > 65536 || n > 1048576 || k % 128 || n % 128 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
        input->ne[0] != k || !rows64 || rows64 > 1024 || output->ne[0] != n || output->type != HTP_TYPE_F32 ||
        (input->type != HTP_TYPE_F32 && input->type != HTP_TYPE_F16)) return HTP_STATUS_INVAL_PARAMS;
    const unsigned rows = (unsigned) rows64, element = input->type == HTP_TYPE_F32 ? 4 : 2;
    if ((uint64_t) (k / 128) * (n / 128) * exl3_group_bytes(bits) > weight->size ||
        input->nb[0] != element || output->nb[0] != 4 ||
        offset_row(input, rows - 1) + (uint64_t) k * element > input->size ||
        offset_row(output, rows - 1) + (uint64_t) n * 4 > output->size) return HTP_STATUS_INVAL_PARAMS;
    const struct htp_mm_kernel_params * params = (const struct htp_mm_kernel_params *) octx->kernel_params;
    if (params->pipeline > 1) return HTP_STATUS_INVAL_PARAMS;
    if (!htp_ops_context_set_n_threads(octx, params->n_threads)) return HTP_STATUS_INVAL_PARAMS;
    const bool hmx = params->pipeline != 0;
    if (hmx && !octx->ctx->hmx_enabled) return HTP_STATUS_NO_SUPPORT;
    const size_t extra = hmx ? exl3_hmx_extra(k, octx->n_threads) : 0;
    if (extra >= octx->ctx->vtcm_size) return HTP_STATUS_VTCM_TOO_SMALL;
    const size_t capacity = exl3_row_chunk(bits, k, octx->n_threads, octx->ctx->vtcm_size - extra);
    if (!capacity) return HTP_STATUS_VTCM_TOO_SMALL;
    const unsigned chunk = rows < capacity ? rows : (unsigned) capacity;
    const size_t per_thread = hmx ? exl3_hmx_thread_bytes(bits, chunk) : exl3_thread_bytes(bits, chunk);
    const size_t prefix = hmx ? exl3_hmx_prefix_bytes(k, chunk) : (size_t) chunk * k * 4;
    const size_t total = prefix + per_thread * octx->n_threads;
    if (total > octx->ctx->vtcm_size) return HTP_STATUS_VTCM_TOO_SMALL;
    float * xh = (float *) octx->ctx->vtcm_base;
    uint16_t * act_half = (uint16_t *) (octx->ctx->vtcm_base + (hmx ? exl3_hmx_activation_offset(k, chunk) : (size_t) chunk * k * 4));
    uint16_t * scales = act_half + k * 32u;
    unsigned char * scratch = octx->ctx->vtcm_base + prefix;
    qurt_mutex_t hmx_lock = QURT_MUTEX_INIT;
    for (unsigned first_row = 0; first_row < rows; first_row += chunk) {
        const unsigned active_rows = rows - first_row < chunk ? rows - first_row : chunk;
        for (unsigned kb = 0; kb < k / 128; ++kb) {
            const unsigned char * group = (const unsigned char *) (uintptr_t) weight->data + kb * exl3_group_bytes(bits);
            const uint16_t * su = exl3_group_su(group, bits);
            for (unsigned row = 0; row < active_rows; ++row) {
                const unsigned char * src = (const unsigned char *) (uintptr_t) input->data + offset_row(input, first_row + row);
                for (unsigned j = 0; j < 128; ++j) {
                    float value;
                    if (element == 4) memcpy(&value, src + (kb * 128 + j) * 4, 4);
                    else { uint16_t h; memcpy(&h, src + (kb * 128 + j) * 2, 2); value = half_value(&h); }
                    xh[row * k + kb * 128 + j] = value * half_value(su + j);
                }
                had128(xh + row * k + kb * 128);
            }
        }
        if (hmx) {
            memset(act_half, 0, k * 64u);
            hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
            for (unsigned row = 0; row < active_rows; row += 2) for (unsigned col = 0; col < k; col += 64) {
                const float * a0 = xh + row * k + col;
                const HVX_Vector v0 = hvx_vec_f32_to_f16(hvx_vmem(a0), hvx_vmem(a0 + 32));
                HVX_Vector v1 = Q6_V_vzero();
                if (row + 1 < active_rows) {
                    const float * a1 = a0 + k;
                    v1 = hvx_vec_f32_to_f16(hvx_vmem(a1), hvx_vmem(a1 + 32));
                }
                const HVX_VectorPair pair = Q6_W_vshuff_VVR(v1, v0, -2);
                hvx_vmem(act_half + (col / 32) * 1024 + (row / 2) * 64) = Q6_V_lo_W(pair);
                hvx_vmem(act_half + (col / 32 + 1) * 1024 + (row / 2) * 64) = Q6_V_hi_W(pair);
            }
        }
        struct exl3_task task = {octx, bits, k, n, active_rows, first_row, per_thread, xh, scratch, hmx, act_half, scales, &hmx_lock};
        if (!work_queue_run(octx->ctx->work_queue, exl3_worker, &task, octx->n_threads)) return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}
