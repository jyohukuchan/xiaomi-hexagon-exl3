#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "hex-utils.h"
#include "hvx-base.h"
#include "hvx-copy.h"
#include "matmul-ops.h"
#include "exl3-block.h"
#include "hexagon/exl3-hvx.h"
#include <math.h>
#include <string.h>

static uint16_t mul1_table[65536];
static qurt_mutex_t table_mutex = QURT_MUTEX_INIT;
static int table_ready;

static void prepare_table(void) {
    qurt_mutex_lock(&table_mutex);
    if (!table_ready) {
        const uint16_t invbits = 0x1eee, biasbits = 0xc931;
        _Float16 inv, bias;
        memcpy(&inv, &invbits, 2); memcpy(&bias, &biasbits, 2);
        // SDK 19 FP16 autovectorization changes codebook values; initialize once in scalar code.
        #pragma clang loop vectorize(disable) interleave(disable)
        for (unsigned state = 0; state < 65536; ++state) {
            const uint32_t product = state * 0x83dcd12du;
            const unsigned sum = (product & 255) + ((product >> 8) & 255) + ((product >> 16) & 255) + (product >> 24);
            const uint16_t hbits = (uint16_t) (0x6400u + sum);
            _Float16 h;
            memcpy(&h, &hbits, 2);
            h = (_Float16) fmaf((float) h, (float) inv, (float) bias);
            memcpy(mul1_table + state, &h, 2);
        }
        table_ready = 1;
    }
    qurt_mutex_unlock(&table_mutex);
}

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
    uint16_t * table;
    float * xh;
    unsigned char * scratch;
};

static void exl3_worker(unsigned nth, unsigned ith, void * opaque) {
    struct exl3_task * task = (struct exl3_task *) opaque;
    const struct htp_tensor * weight = task->octx->src[0];
    const struct htp_tensor * output = task->octx->dst;
    unsigned char * local = task->scratch + ith * task->per_thread;
    uint16_t * decoded = (uint16_t *) local;
    float * result = (float *) (decoded + 128 * 128);
    uint16_t * packed = (uint16_t *) (result + task->rows * 128);
    HVX_Vector * gathered = (HVX_Vector *) ((unsigned char *) packed + exl3_group_bytes(task->bits));
    uint16_t * tile = (uint16_t *) (gathered + 1);
    for (unsigned nb = ith; nb < task->n / 128; nb += nth) {
        memset(result, 0, task->rows * 128 * 4);
        for (unsigned kb = 0; kb < task->k / 128; ++kb) {
            const unsigned char * group = (const unsigned char *) (uintptr_t) weight->data +
                ((size_t) nb * (task->k / 128) + kb) * exl3_group_bytes(task->bits);
            memcpy(packed, group, exl3_group_bytes(task->bits));
            for (unsigned r = 0; r < 8; ++r) for (unsigned c = 0; c < 8; ++c) {
                exl3_hvx_decode_tile(packed + (r * 8 + c) * 16 * task->bits, task->bits, task->table, gathered, tile);
                for (unsigned j = 0; j < 16; ++j) memcpy(decoded + (r * 16 + j) * 128 + c * 16, tile + j * 16, 32);
            }
            for (unsigned i = 0; i < 128; ++i) {
                HVX_VectorPair w0 = hvx_vec_f16_to_f32(*(HVX_Vector *) (decoded + i * 128));
                HVX_VectorPair w1 = hvx_vec_f16_to_f32(*(HVX_Vector *) (decoded + i * 128 + 64));
                HVX_Vector w[4] = {Q6_V_lo_W(w0), Q6_V_hi_W(w0), Q6_V_lo_W(w1), Q6_V_hi_W(w1)};
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
    if (!htp_ops_context_set_n_threads(octx, params->n_threads)) return HTP_STATUS_INVAL_PARAMS;
    const size_t capacity = exl3_row_chunk(bits, k, octx->n_threads, octx->ctx->vtcm_size);
    if (!capacity) return HTP_STATUS_VTCM_TOO_SMALL;
    const unsigned chunk = rows < capacity ? rows : (unsigned) capacity;
    const size_t per_thread = 32768 + (size_t) chunk * 512 + exl3_group_bytes(bits) + 128 + 512;
    const size_t total = 131072 + (size_t) chunk * k * 4 + per_thread * octx->n_threads;
    if (total > octx->ctx->vtcm_size) return HTP_STATUS_VTCM_TOO_SMALL;
    prepare_table();
    uint16_t * table = (uint16_t *) octx->ctx->vtcm_base;
    memcpy(table, mul1_table, 131072);
    float * xh = (float *) (table + 65536);
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
        struct exl3_task task = {octx, bits, k, n, active_rows, first_row, per_thread, table, xh, (unsigned char *) (xh + (size_t) chunk * k)};
        if (!work_queue_run(octx->ctx->work_queue, exl3_worker, &task, octx->n_threads)) return HTP_STATUS_INTERNAL_ERR;
    }
    return HTP_STATUS_OK;
}
