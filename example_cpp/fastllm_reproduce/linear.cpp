#include "computeutils.h"
#include "utils.h"

MultiThreadLinearInt8Int4GroupOp::MultiThreadLinearInt8Int4GroupOp(uint8_t *a,
                                                                   uint8_t *b,
                                                                   float *c,
                                                                   int n,
                                                                   int m,
                                                                   int k,
                                                                   int kstride,
                                                                   int *weightSums,
                                                                   float *weightMins,
                                                                   float *scales,
                                                                   float *bias,
                                                                   float *iscales,
                                                                   float *izeros,
                                                                   float *inputSums,
                                                                   int group,
                                                                   int groupCnt) {
    this->a = a; // int8 激活 (n × k)
    this->b = b; // int4 权重 (k × m) 每字节 2 值
    this->c = c; // float 输出 (n × m)
    this->n = n;
    this->m = m;
    this->k = k;
    this->kstride = kstride;
    this->weightSums = weightSums;
    this->weightMins = weightMins;
    this->scales = scales;
    this->bias = bias;
    this->iscales = iscales;
    this->izeros = izeros;
    this->inputSums = inputSums;
    this->group = group;
    this->groupCnt = groupCnt;
}

void MultiThreadLinearInt8Int4GroupOp::Run() {}

MultiThreadLinearFloat32Float32Op::MultiThreadLinearFloat32Float32Op(
    float *inputData, float *weightData, float *biasData, float *outputData, int n, int m, int k, int st, int end) {
    this->inputData = inputData;
    this->weightData = weightData;
    this->biasData = biasData;
    this->outputData = outputData;
    this->n = n;
    this->m = m;
    this->k = k;
    this->st = st;
    this->end = end;
}

void MultiThreadLinearFloat32Float32Op::Run() {
    for (int i = 0; i < n; i++) {
        for (int j = st; j < end; j++) {
            float now = biasData ? biasData[j] : 0.0f;
            int l = 0;
#ifdef __aarch64__
            float32x4_t sum = {0, 0, 0, 0};
            for (; l + 3 < m; l += 4) {
                sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(inputData + i * m + l), vld1q_f32(weightData + j * m + l)));
            }
            now += sum[0] + sum[1] + sum[2] + sum[3];
#else
#ifdef __AVX2__
            __m256 vsum = _mm256_setzero_ps();
            for (; l + 7 < m; l += 8) {
                __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                __m256 vw = _mm256_loadu_ps(weightData + j * m + l);
                vsum = _mm256_fmadd_ps(vi, vw, vsum);
            }
            now += Floatsum(vsum);
#endif
#endif
            for (; l < m; l++) {
                now += inputData[i * m + l] * weightData[j * m + l];
            }
            outputData[i * k + j] = now;
        }
    }
}

MultiThreadLinearFloat32Float16Op::MultiThreadLinearFloat32Float16Op(
    float *inputData, uint16_t *weightData, float *biasData, float *outputData, int n, int m, int k, int st, int end) {
    this->inputData = inputData;
    this->weightData = weightData;
    this->biasData = biasData;
    this->outputData = outputData;
    this->n = n;
    this->m = m;
    this->k = k;
    this->st = st;
    this->end = end;
}

void MultiThreadLinearFloat32Float16Op::Run() {
    for (int i = 0; i < n; i++) {
        for (int j = st; j < end; j++) {
            float now = biasData ? biasData[j] : 0.0f;
            int l = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
            for (; l + 7 < m; l += 8) {
                sum = vfmaq_f16(sum, vld1q_f16((float16_t *)inputData + i * m + l), vld1q_f16((float16_t *)weightData + j * m + l));
            }
            now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#else
#ifdef __aarch64__
            float32x4_t sum = {0, 0, 0, 0};
            for (; l + 3 < m; l += 4) {
                float32x4_t vcur = {g_fp16ToFp32Manager.dict[weightData[j * m + l]],
                                    g_fp16ToFp32Manager.dict[weightData[j * m + l + 1]],
                                    g_fp16ToFp32Manager.dict[weightData[j * m + l + 2]],
                                    g_fp16ToFp32Manager.dict[weightData[j * m + l + 3]]};
                sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(inputData + i * m + l), vcur));
            }
            now += sum[0] + sum[1] + sum[2] + sum[3];
#else
#ifdef __AVX2__
            __m256 vsum = _mm256_setzero_ps();
            for (; l + 7 < m; l += 8) {
                __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                __m256 vw = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(weightData + j * m + l)));
                vsum = _mm256_fmadd_ps(vi, vw, vsum);
            }
            now += Floatsum(vsum);
#endif
#endif
#endif
            for (; l < m; l++) {
                now += inputData[i * m + l] * g_fp16ToFp32Manager.dict[weightData[j * m + l]];
            }
            outputData[i * k + j] = now;
        }
    }
}

void RunLinearFloat32Float32(float *inputData,
                             float *weightData,
                             float *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum) {
    int per = k / threadNum;
    int cur = 0;
    std::vector<MultiThreadLinearFloat32Float32Op *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = cur + per + (cur + per * (threadNum - i) < k);
        ops.push_back(new MultiThreadLinearFloat32Float32Op(inputData, weightData, biasData, outputData, n, m, k, cur, end));
        cur = end;
    }
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(startTid + i, ops[i]);
    }
    for (int i = 0; i < threadNum; i++) {
        pool->Wait(startTid + i);
        delete ops[i];
    }
}
