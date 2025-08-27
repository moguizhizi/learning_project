#include "computeutils.h"

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
