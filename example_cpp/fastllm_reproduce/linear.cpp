#include "common_class.h"
#include "computeutils.h"
#include "cpudevice.h"
#include "utils.h"
#include <cstring>

MultiThreadLinearBFloat16FP8E4M3Op::MultiThreadLinearBFloat16FP8E4M3Op(uint16_t *inputData,
                                                                       uint8_t *weightData,
                                                                       float *biasData,
                                                                       float *outputData,
                                                                       int n,
                                                                       int m,
                                                                       int k,
                                                                       int st,
                                                                       int end,
                                                                       float *scales,
                                                                       int blockK,
                                                                       int blockM) {}

void MultiThreadLinearBFloat16FP8E4M3Op::Run() {}

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

void MultiThreadLinearInt8Int4GroupOp::Run() {
#ifdef __AVX2__
    if (group == 1) {
        int block = 0;
        int realGroup = (m - 1) / groupCnt + 1;
        std::vector<float> tempValue, values;
        tempValue.resize(n);
        values.resize(n * k);
        for (; block < n; block++) {
            tempValue[block] = (inputSums[block] - izeros[block] * groupCnt) * iscales[block];
        }

        if (cpuInstructInfo.hasAVX512VNNI && MatMulInt8Int4_AVX512VNNI(a, b, values.data(), n, m, k)) {
        } else {
            block = 0;
            for (; block + 3 < n; block += 4) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    uint8_t *a = weightWalk + (i * m) / 2;
                    uint8_t *b = inputStart;

                    __m256i acc0 = _mm256_setzero_si256();
                    __m256i acc1 = _mm256_setzero_si256();
                    __m256i acc2 = _mm256_setzero_si256();
                    __m256i acc3 = _mm256_setzero_si256();

                    const __m256i lowMask = _mm256_set1_epi8(0xf);
                    const __m256i ones = _mm256_set1_epi16(1);
                    int j = 0, ans = 0;
                    for (; j + 31 < m; j += 32) {
                        __m128i orix = _mm_loadu_si128((const __m128i *)(a + j / 2));
                        __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                        __m256i bx = _mm256_and_si256(lowMask, bytex);
                        __m256i by0 = _mm256_loadu_si256((const __m256i *)(b + j));
                        __m256i by1 = _mm256_loadu_si256((const __m256i *)(b + m * 1 + j));
                        __m256i by2 = _mm256_loadu_si256((const __m256i *)(b + m * 2 + j));
                        __m256i by3 = _mm256_loadu_si256((const __m256i *)(b + m * 3 + j));

                        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(by0, bx), ones));
                        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(by1, bx), ones));
                        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(by2, bx), ones));
                        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(by3, bx), ones));
                    }
                    values[block * k + i] = I32sum(acc0);
                    values[(block + 1) * k + i] = I32sum(acc1);
                    values[(block + 2) * k + i] = I32sum(acc2);
                    values[(block + 3) * k + i] = I32sum(acc3);
                }
            }

            for (; block + 2 < n; block += 3) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    uint8_t *a = weightWalk + (i * m) / 2;
                    uint8_t *b = inputStart;

                    __m256i acc0 = _mm256_setzero_si256();
                    __m256i acc1 = _mm256_setzero_si256();
                    __m256i acc2 = _mm256_setzero_si256();

                    const __m256i lowMask = _mm256_set1_epi8(0xf);
                    const __m256i ones = _mm256_set1_epi16(1);
                    int j = 0, ans = 0;
                    for (; j + 31 < m; j += 32) {
                        __m128i orix = _mm_loadu_si128((const __m128i *)(a + j / 2));
                        __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                        __m256i bx = _mm256_and_si256(lowMask, bytex);
                        __m256i by0 = _mm256_loadu_si256((const __m256i *)(b + j));
                        __m256i by1 = _mm256_loadu_si256((const __m256i *)(b + m * 1 + j));
                        __m256i by2 = _mm256_loadu_si256((const __m256i *)(b + m * 2 + j));

                        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(by0, bx), ones));
                        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(by1, bx), ones));
                        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(by2, bx), ones));
                    }
                    values[block * k + i] = I32sum(acc0);
                    values[(block + 1) * k + i] = I32sum(acc1);
                    values[(block + 2) * k + i] = I32sum(acc2);
                }
            }

            for (; block + 1 < n; block += 2) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    uint8_t *a = weightWalk + (i * m) / 2;
                    uint8_t *b = inputStart;

                    __m256i acc0 = _mm256_setzero_si256();
                    __m256i acc1 = _mm256_setzero_si256();

                    const __m256i lowMask = _mm256_set1_epi8(0xf);
                    const __m256i ones = _mm256_set1_epi16(1);
                    int j = 0, ans = 0;
                    for (; j + 31 < m; j += 32) {
                        __m128i orix = _mm_loadu_si128((const __m128i *)(a + j / 2));
                        __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                        __m256i bx = _mm256_and_si256(lowMask, bytex);
                        __m256i by0 = _mm256_loadu_si256((const __m256i *)(b + j));
                        __m256i by1 = _mm256_loadu_si256((const __m256i *)(b + m * 1 + j));

                        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(by0, bx), ones));
                        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(by1, bx), ones));
                    }
                    values[block * k + i] = I32sum(acc0);
                    values[(block + 1) * k + i] = I32sum(acc1);
                }
            }

            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    uint8_t *a = weightWalk + (i * m) / 2;
                    uint8_t *b = inputStart;

                    __m256i acc = _mm256_setzero_si256();
                    const __m256i lowMask = _mm256_set1_epi8(0xf);
                    const __m256i ones = _mm256_set1_epi16(1);
                    int j = 0, ans = 0;
                    for (; j + 31 < m; j += 32) {
                        __m128i orix = _mm_loadu_si128((const __m128i *)(a + j / 2));
                        __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                        __m256i bx = _mm256_and_si256(lowMask, bytex);
                        __m256i by = _mm256_loadu_si256((const __m256i *)(b + j));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(by, bx), ones));
                    }
                    values[block * k + i] = I32sum(acc);
                }
            }
        }

        block = 0;
        for (; block < n; block++) {
            int i = 0;
            for (; i < k; i++) {
                const float vv = (float)values[block * k + i] - weightSums[i] * izeros[block];
                float sum = scales[i] * iscales[block] * vv + weightMins[i] * tempValue[block];
                ((float *)c)[block * kstride + i] = sum + (bias == nullptr ? 0.0 : bias[i]);
            }
        }
        return;
    }
#endif
    std::vector<float> values;
    values.resize(group);

    int block = 0;
    int realGroup = (m - 1) / groupCnt + 1;
    std::vector<float> tempValue;
    tempValue.resize(realGroup);
    for (; block < n; block++) {
        for (int g = 0; g < realGroup; g++) {
            int iid = block * group + g;
            tempValue[g] = (inputSums[iid] - izeros[iid] * groupCnt) * iscales[iid];
        }

        uint8_t *weightWalk = b;
        uint8_t *inputStart = a + block * m;

        for (int i = 0; i < k; i++) {
            std::fill(values.begin(), values.end(), 0.0f);
            uint8_t *inputWalk = inputStart;
            float sum = 0.0;

            for (int g = 0; g < realGroup; g++) {
                int st = g * groupCnt, end = std::min(m, (g + 1) * groupCnt);
                float &value = values[g];
                int j = st;
#ifdef __ARM_FEATURE_DOTPROD
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                uint32x2_t sum0 = {0, 0};

                for (; j + 15 < end; j += 16) {
                    uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                    uint8x8x2_t in = vld2_u8(inputWalk + j);
                    uint8x8_t va = vand_u8(ori, maskLow);
                    uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                    sum0 = vdot_u32(sum0, va, in.val[1]);
                    sum0 = vdot_u32(sum0, vb, in.val[0]);
                }
                value += sum0[0] + sum0[1];
#elif defined(__aarch64__)
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                uint32x4_t sum0 = {0, 0, 0, 0};

                for (; j + 15 < end; j += 16) {
                    uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                    uint8x8x2_t in = vld2_u8(inputWalk + j);
                    uint8x8_t va = vand_u8(ori, maskLow);
                    uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                    sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                    sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
                }
                value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
                value += DotU4U8(weightWalk + (i * m + st) / 2, inputWalk + st, end - st);
                j += (end - st);
#endif
                for (; j + 1 < end; j += 2) {
                    int id = (i * m + j) / 2;
                    value += (weightWalk[id] >> 4) * inputWalk[j];
                    value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
                }
            }

            int g = 0;
#ifdef __aarch64__
            float32x4_t vSum = vdupq_n_f32(0.0f);
            float32x4_t vGroupCnt = vdupq_n_f32(groupCnt);
            for (; g + 3 < realGroup; g += 4) {
                int iid = block * group + g;
                int gid = i * group + g;
                float32x4_t vValue = vld1q_f32(values.data() + g);
                float32x4_t vWeightSum = vcvtq_f32_s32(vld1q_s32(weightSums + gid));
                float32x4_t vWeightMin = vld1q_f32(weightMins + gid);
                float32x4_t vScale = vld1q_f32(scales + gid);
                float32x4_t vIzero = vld1q_f32(izeros + iid);
                float32x4_t vIscale = vld1q_f32(iscales + iid);
                float32x4_t vInputSum = vld1q_f32(inputSums + iid);
                float32x4_t vMiddle = vsubq_f32(vInputSum, vmulq_f32(vIzero, vGroupCnt));
                vValue = vsubq_f32(vValue, vmulq_f32(vWeightSum, vIzero));
                vSum = vaddq_f32(vSum, vmulq_f32(vScale, vmulq_f32(vIscale, vValue)));
                vSum = vaddq_f32(vSum, vmulq_f32(vWeightMin, vmulq_f32(vMiddle, vIscale)));
            }
            sum += vSum[0] + vSum[1] + vSum[2] + vSum[3];
#endif
            // 处理剩余元素（标量处理）
            for (; g < realGroup; g++) {
                const int iid = block * group + g;
                const int gid = i * group + g;

                // 修正value为float类型
                const float value = (float)values[g] - weightSums[gid] * izeros[iid];
                sum += scales[gid] * iscales[iid] * value + weightMins[gid] * tempValue[g];
            }

            if (group * groupCnt > m) {
                int iid = block * group + group - 1;
                int gid = i * group + group - 1;
                sum += weightMins[gid] * izeros[iid] * (group * groupCnt - m) * iscales[iid];
            }

            ((float *)c)[block * kstride + i] = sum + (bias == nullptr ? 0.0 : bias[i]);
        }
    }
}

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

// a = [n, m], b = [k, m], c = aT(b') = [n, k]
void RunLinearInt8Int4Group(uint8_t *a,
                            uint8_t *b,
                            float *c,
                            int n,
                            int m,
                            int k,
                            int group,
                            int groupCnt,
                            int *weightSums,
                            float *weightMins,
                            float *scales,
                            float *bias,
                            float *inputSums,
                            float *iscales,
                            float *izeros,
                            AliveThreadPool *pool,
                            int startTid,
                            int threadNum) {
    int per = k / threadNum;
    int cur = 0;
    std::vector<MultiThreadLinearInt8Int4GroupOp *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
        ops.push_back(new MultiThreadLinearInt8Int4GroupOp(a,
                                                           b + cur * m / 2,
                                                           c + cur,
                                                           n,
                                                           m,
                                                           end - cur,
                                                           k,
                                                           weightSums + cur * group,
                                                           weightMins + cur * group,
                                                           scales + cur * group,
                                                           (bias == nullptr ? (float *)nullptr : bias + cur),
                                                           iscales,
                                                           izeros,
                                                           inputSums,
                                                           group,
                                                           groupCnt));
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

void RunLinearFloat32Int4Group(float *inputData,
                               Data &weight,
                               float *outputData,
                               float *biasData,
                               int n,
                               int m,
                               int k,
                               int group,
                               int groupCnt,
                               AliveThreadPool *pool,
                               int startTid,
                               int threadNum) {
    weight.CalcWeightSum();
    std::vector<LowBitConfig> inputConfigs;
    std::vector<uint8_t> uinput;
    std::vector<float> inputSums, iscales, izeros;
    OnlineQuantization(inputData, uinput, inputConfigs, n, m, group, groupCnt, inputSums, iscales, izeros, 1);
    RunLinearInt8Int4Group(uinput.data(),
                           (uint8_t *)weight.cpuData,
                           outputData,
                           n,
                           m,
                           k,
                           group,
                           groupCnt,
                           weight.weightSum.data(),
                           weight.mins.data(),
                           weight.scales.data(),
                           biasData,
                           inputSums.data(),
                           iscales.data(),
                           izeros.data(),
                           pool,
                           startTid,
                           threadNum);
}

void RunLinearFloat32Float16(float *inputData,
                             uint16_t *weightData,
                             float *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    uint16_t *temp = new uint16_t[n * m];
    for (int i = 0; i < n * m; i++) {
        temp[i] = float_to_half(inputData[i]);
    }
    inputData = (float *)temp;
#endif
    int per = k / threadNum;
    int cur = 0;
    std::vector<MultiThreadLinearFloat32Float16Op *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = cur + per + (cur + per * (threadNum - i) < k);
        if (i == threadNum - 1) {
            end = k;
        }
        ops.push_back(new MultiThreadLinearFloat32Float16Op(inputData, weightData, biasData, outputData, n, m, k, cur, end));
        cur = end;
    }
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(startTid + i, ops[i]);
    }
    for (int i = 0; i < threadNum; i++) {
        pool->Wait(startTid + i);
        delete ops[i];
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    delete[] temp;
#endif
}

void MatMulInt8Int8(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride) {
#ifdef __ARM_FEATURE_DOTPROD
#define RUNBLOCK(x)                                                                                                                                  \
    for (; block + (x - 1) < n; block += (x))                                                                                                        \
        MatMulInt8Int8RunSomeBlock(b, a + block * m, c, (x), sum, vi, block, k, m, kstride);
    int block = 0;
    uint32x4_t sum[16];
    uint8x16_t vi[16];
    RUNBLOCK(16);
    RUNBLOCK(8);
    RUNBLOCK(7);
    RUNBLOCK(6);
    RUNBLOCK(5);
    RUNBLOCK(4);
    RUNBLOCK(3);
    RUNBLOCK(2);
    RUNBLOCK(1);
#undef RUNBLOCK
#elif defined(__aarch64__)
    int block = 0;
    for (; block < n; block++) {
        uint8_t *weightWalk = b;
        uint8_t *inputStart = a + block * m;

        for (int i = 0; i < k; i++) {
            int value = 0;
            uint8_t *inputWalk = inputStart;

            int per = 64;
            int cnt = m / per;
            int sur = m % per;

            uint32x4_t sum = {0};
            uint16x8_t temp = {0};
            uint16x8_t temp1 = {0};
            uint16x8_t temp2 = {0};
            uint16x8_t temp3 = {0};
            uint16x8_t temp4 = {0};
            uint16x8_t temp5 = {0};
            uint16x8_t temp6 = {0};
            uint16x8_t temp7 = {0};

            while (cnt--) {
                temp = vmull_u8(vld1_u8(inputWalk), vld1_u8(weightWalk));
                temp1 = vmull_u8(vld1_u8(inputWalk + 8), vld1_u8(weightWalk + 8));
                temp2 = vmull_u8(vld1_u8(inputWalk + 16), vld1_u8(weightWalk + 16));
                temp3 = vmull_u8(vld1_u8(inputWalk + 24), vld1_u8(weightWalk + 24));
                temp4 = vmull_u8(vld1_u8(inputWalk + 32), vld1_u8(weightWalk + 32));
                temp5 = vmull_u8(vld1_u8(inputWalk + 40), vld1_u8(weightWalk + 40));
                temp6 = vmull_u8(vld1_u8(inputWalk + 48), vld1_u8(weightWalk + 48));
                temp7 = vmull_u8(vld1_u8(inputWalk + 56), vld1_u8(weightWalk + 56));

                sum = vpadalq_u16(sum, temp);
                sum = vpadalq_u16(sum, temp1);
                sum = vpadalq_u16(sum, temp2);
                sum = vpadalq_u16(sum, temp3);
                sum = vpadalq_u16(sum, temp4);
                sum = vpadalq_u16(sum, temp5);
                sum = vpadalq_u16(sum, temp6);
                sum = vpadalq_u16(sum, temp7);

                inputWalk += per;
                weightWalk += per;
            }

            value += (sum[0] + sum[1] + sum[2] + sum[3]);
            while (sur--) {
                value += (int)(*(weightWalk++)) * (*(inputWalk++));
            }

            c[block * kstride + i] = value;
        }
    }
#elif defined(__AVX2__)
    int block = 0;
    for (; block < n; block++) {
        uint8_t *weightWalk = b;
        uint8_t *inputStart = a + block * m;

        for (int i = 0; i < k; i++) {
            uint8_t *inputWalk = inputStart;

            c[block * kstride + i] = DotU8U8(inputWalk, weightWalk, m);
            weightWalk += m;
        }
    }
#else
    int block = 0;
    for (; block < n; block++) {
        uint8_t *weightWalk = b;
        uint8_t *inputStart = a + block * m;

        for (int i = 0; i < k; i++) {
            int value = 0;
            uint8_t *inputWalk = inputStart;
            for (int j = 0; j < m; j++) {
                value += (int)(*(weightWalk++)) * (*(inputWalk++));
            }

            c[block * kstride + i] = value;
        }
    }
#endif
}

MultiThreadLinearInt8Int8Op::MultiThreadLinearInt8Int8Op(uint8_t *a,
                                                         uint8_t *b,
                                                         int32_t *c,
                                                         int n,
                                                         int m,
                                                         int k,
                                                         int kstride,
                                                         int *weightSums,
                                                         int *weightZeros,
                                                         float *scales,
                                                         float *bias,
                                                         float *iscales,
                                                         float *izeros,
                                                         float *inputSums) {
    this->a = a;
    this->b = b;
    this->c = c;
    this->n = n;
    this->m = m;
    this->k = k;
    this->kstride = kstride;
    this->weightSums = weightSums;
    this->weightZeros = weightZeros;
    this->scales = scales;
    this->bias = bias;
    this->iscales = iscales;
    this->izeros = izeros;
    this->inputSums = inputSums;
}

void MultiThreadLinearInt8Int8Op::Run() {
    MatMulInt8Int8(a, b, c, n, m, k, kstride);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float value = ((int32_t *)c)[i * kstride + j];
#ifdef __AVX2__
            value += (128 * weightSums[j]);
            value += (128 * inputSums[i]);
            value -= m * 128 * 128;
#endif
            value -= weightSums[j] * izeros[i];
            value -= inputSums[i] * weightZeros[j];
            value += (int)izeros[i] * weightZeros[j] * m;
            ((float *)c)[i * kstride + j] = scales[j] * iscales[i] * value + (bias == nullptr ? 0.0 : bias[j]);
        }
    }
}

MultiThreadLinearFloat32Int2GroupOp::MultiThreadLinearFloat32Int2GroupOp(
    float *inputData, Data *weight, float *biasData, float *outputData, int n, int m, int k, int st, int end) {
    this->inputData = inputData;
    this->weight = weight;
    this->biasData = biasData;
    this->outputData = outputData;
    this->n = n;
    this->m = m;
    this->k = k;
    this->st = st;
    this->end = end;
}

void MultiThreadLinearFloat32Int2GroupOp::Run() {
    int group = this->weight->group;
    int groupCnt = this->weight->groupCnt;
    std::vector<float> mins = this->weight->mins;
    std::vector<float> scales = this->weight->scales;
    for (int i = 0; i < this->n; i++) {
        for (int j = this->st; j < this->end; j++) {
            float now = biasData ? biasData[j] : 0.0f;
            for (int l = 0; l < this->m; l++) {
                int gid = j * group + l / groupCnt;
                float min = mins[gid];
                float scale = scales[gid];
                uint8_t w = this->weight->cpuData[(j * m + l) / 4];
                w = (w >> ((3 - l % 4) * 2)) & 3;
                now += inputData[i * m + l] * (min + scale * w);
            }
            this->outputData[i * k + j] = now;
        }
    }
}

void RunLinearFloat32Int2Group(float *inputData,
                               Data &weight,
                               float *outputData,
                               float *biasData,
                               int n,
                               int m,
                               int k,
                               int group,
                               int groupCnt,
                               AliveThreadPool *pool,
                               int startTid,
                               int threadNum) {
    int per = k / threadNum;
    int cur = 0;
    std::vector<MultiThreadLinearFloat32Int2GroupOp *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = cur + per + (cur + per * (threadNum - i) < k);
        ops.push_back(new MultiThreadLinearFloat32Int2GroupOp(inputData, &weight, biasData, outputData, n, m, k, cur, end));
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

void RunLinearFloat16Float32(uint16_t *inputData,
                             float *weightData,
                             uint16_t *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum) {
    std::vector<float> floatInput, floatOutput;
    floatInput.resize(n * m);
    floatOutput.resize(n * k);
    Float16ToFloat32(inputData, floatInput.data(), n * m);
    RunLinearFloat32Float32(floatInput.data(), weightData, floatOutput.data(), biasData, n, m, k, pool, startTid, threadNum);
    Float32ToFloat16(floatOutput.data(), outputData, n * k);
}

MultiThreadLinearFloat16Float16Op::MultiThreadLinearFloat16Float16Op(
    uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, int n, int m, int k, int st, int end) {
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

void MultiThreadLinearFloat16Float16Op::Run() { MatMulFloat16Float16(inputData, weightData, biasData, outputData, n, m, k, st, end); }

void MatMulFloat16Float16(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, int n, int m, int k, int st, int end) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (n > 1 && n < 8) {
        const int BLOCKA = 7;
        float16x8_t va[BLOCKA], vb, vc[BLOCKA];

        for (int i = 0; i < n; i += BLOCKA) {
            int cur = std::min(BLOCKA, n - i);
            for (int j = st; j < end; j++) {
                for (int l1 = 0; l1 < cur; l1++) {
                    vc[l1] = vdupq_n_f16(0.0f);
                }

                for (int k = 0; k < m; k += 8) {
                    for (int l = 0; l < cur; l++) {
                        va[l] = vld1q_f16((float16_t *)inputData + (i + l) * m + k);
                    }
                    vb = vld1q_f16((float16_t *)weightData + j * m + k);

                    for (int l1 = 0; l1 < cur; l1++) {
                        vc[l1] = vfmaq_f16(vc[l1], va[l1], vb);
                    }
                }

                for (int l0 = 0; l0 < cur; l0++) {
                    float now = vc[l0][0] + vc[l0][1] + vc[l0][2] + vc[l0][3] + vc[l0][4] + vc[l0][5] + vc[l0][6] + vc[l0][7];
                    if (biasData != nullptr) {
                        now += biasData[j];
                    }
                    outputData[(i + l0) * k + j] = float_to_half(now);
                }
            }
        }
    } else if (n > 1) {
        const int BN = 64, BM = 64, BK = 64;
        uint16_t *a = new uint16_t[BN * BM];
        uint16_t *b = new uint16_t[BK * BM];
        uint16_t *c = new uint16_t[BN * BK];
        if (biasData == nullptr) {
            for (int i = 0; i < n; i++) {
                memset(outputData + i * k + st, 0, (end - st) * sizeof(uint16_t));
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    outputData[i * k + j] = float_to_half(biasData[j]);
                }
            }
        }

        for (int mst = 0; mst < m; mst += BM) {
            int mend = std::min(mst + BM, m);
            memset(c, 0, BN * BK * sizeof(uint16_t));
            for (int nst = 0; nst < n; nst += BN) {
                int nend = std::min(nst + BN, n);
                for (int i = nst; i < nend; i++) {
                    memcpy(a + (i - nst) * BM, inputData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                }
                for (int kst = st; kst < end; kst += BK) {
                    int kend = std::min(kst + BK, end);
                    for (int i = kst; i < kend; i++) {
                        memcpy(b + (i - kst) * BM, weightData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                    }

                    const int BLOCKB = 8;
                    float16x8_t va, vb[BLOCKB], vc[BLOCKB];

                    for (int i = 0; i < BN && i < (nend - nst); i++) {
                        for (int j = 0; j < BK; j += BLOCKB) {
                            for (int l1 = 0; l1 < BLOCKB; l1++) {
                                vc[l1] = vdupq_n_f16(0.0f);
                            }

                            for (int k = 0; k < BM; k += 8) {
                                va = vld1q_f16((float16_t *)a + i * BM + k);
                                for (int l = 0; l < BLOCKB; l++) {
                                    vb[l] = vld1q_f16((float16_t *)b + (j + l) * BM + k);
                                }

                                for (int l1 = 0; l1 < BLOCKB; l1++) {
                                    vc[l1] = vfmaq_f16(vc[l1], va, vb[l1]);
                                }
                            }

                            float16x8x2_t temp0 = vtrnq_f16(vc[0], vc[1]);
                            float16x8x2_t temp1 = vtrnq_f16(vc[2], vc[3]);
                            float16x8x2_t temp2 = vtrnq_f16(vc[4], vc[5]);
                            float16x8x2_t temp3 = vtrnq_f16(vc[6], vc[7]);

                            vc[0] = vaddq_f16(temp0.val[0], temp0.val[1]);
                            vc[1] = vaddq_f16(temp1.val[0], temp1.val[1]);
                            vc[2] = vaddq_f16(temp2.val[0], temp2.val[1]);
                            vc[3] = vaddq_f16(temp3.val[0], temp3.val[1]);

                            float32x4x2_t temp4 = vtrnq_f32(vreinterpretq_f32_f16(vc[0]), vreinterpretq_f32_f16(vc[1]));
                            float32x4x2_t temp5 = vtrnq_f32(vreinterpretq_f32_f16(vc[2]), vreinterpretq_f32_f16(vc[3]));

                            vc[0] = vaddq_f16(vreinterpretq_f16_f32(temp4.val[0]), vreinterpretq_f16_f32(temp4.val[1]));
                            vc[1] = vaddq_f16(vreinterpretq_f16_f32(temp5.val[0]), vreinterpretq_f16_f32(temp5.val[1]));

                            vst1q_f16((float16_t *)c + i * BK + j,
                                      vcombine_f16(vadd_f16(vget_high_f16(vc[0]), vget_low_f16(vc[0])),
                                                   vadd_f16(vget_high_f16(vc[1]), vget_low_f16(vc[1]))));

                            /*for (int l0 = 0; l0 < BLOCKA; l0++) {
                                    for (int l1 = 0; l1 < BLOCKB; l1++) {
                                        float now = vc[l0][l1][0] + vc[l0][l1][1] + vc[l0][l1][2] + vc[l0][l1][3] +
                                            vc[l0][l1][4] + vc[l0][l1][5] + vc[l0][l1][6] + vc[l0][l1][7];
                                        c[(i + l0) * block + (j + l1)] = float_to_half(now);
                                    }
                            }*/
                        }
                    }
                    /*
                                                for (int i = 0; i < block && i < (nend - nst); i++) {
                                                    for (int j = 0; j < block; j++) {
                                                        float now = 0.0;
                                                        int k = 0;
                    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                        float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
                                                        for (; k + 7 < block; k += 8) {
                                                            sum = vfmaq_f16(sum, vld1q_f16((float16_t*)a + i * block + k),
                                                                                vld1q_f16((float16_t*)b + j * block + k));
                                                        }
                                                        now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
                    #endif
                                                        for (; k < block; k++) {
                                                            now += fp16tofp32.dict[a[i * block + k]] * fp16tofp32.dict[b[j * block + k]];
                                                        }
                                                        c[i * block + j] = float_to_half(now);
                                                    }
                                                }
                    */

                    for (int i = nst; i < nend; i++) {
                        int j = kst;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        for (; j + 7 < kend; j += 8) {
                            vst1q_f16(
                                (float16_t *)outputData + i * k + j,
                                vaddq_f16(vld1q_f16((float16_t *)outputData + i * k + j), vld1q_f16((float16_t *)c + (i - nst) * BK + (j - kst))));
                        }
#endif
                        for (; j < kend; j++) {
                            outputData[i * k + j] =
                                (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + fp16tofp32.dict[c[(i - nst) * BK + (j - kst)]]);
                        }
                    }
                }
            }
        }
        delete[] a;
        delete[] b;
        delete[] c;
        /*
                        for (int i = 0; i < n; i++) {
                            for (int j = st; j < end; j++) {
                                float now = biasData ? biasData[j] : 0.0f;
                                for (int l = 0; l < m; l++) {
                                    now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                                }
        {
            float a = half_to_float(outputData[i * k + j]);
            if (st == 0 && fabs(a - now) > 1e-1) {
                printf("wrong %d %d %f %f\n", i, j, a, now);
                exit(0);
            }
        }
                                outputData[i * k + j] = float_to_half(now);
                            }
                        }
        */
    } else {
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
#endif
                for (; l < m; l++) {
                    now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = float_to_half(now);
            }
        }
    }
#elif defined(__AVX2__)
    if (n > 8) {
        const int BN = 64, BM = 128, BK = 64;
        uint16_t *a = new uint16_t[BN * BM];
        uint16_t *b = new uint16_t[BK * BM];
        uint16_t *c = new uint16_t[BN * BK];
        if (biasData == nullptr) {
            for (int i = 0; i < n; i++) {
                memset(outputData + i * k + st, 0, (end - st) * sizeof(uint16_t));
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    outputData[i * k + j] = float_to_half(biasData[j]);
                }
            }
        }

        for (int mst = 0; mst < m; mst += BM) {
            int mend = std::min(mst + BM, m);
            memset(c, 0, BN * BK * sizeof(uint16_t));
            for (int nst = 0; nst < n; nst += BN) {
                int nend = std::min(nst + BN, n);
                for (int i = nst; i < nend; i++) {
                    memcpy(a + (i - nst) * BM, inputData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                }
                for (int kst = st; kst < end; kst += BK) {
                    int kend = std::min(kst + BK, end);
                    for (int i = kst; i < kend; i++) {
                        memcpy(b + (i - kst) * BM, weightData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                    }

                    const int BLOCKB = 8;
                    __m256 va, vb[BLOCKB], vc[BLOCKB];

                    for (int i = 0; i < BN && i < (nend - nst); i++) {
                        for (int j = 0; j < BK; j += BLOCKB) {
                            for (int l1 = 0; l1 < BLOCKB; l1++) {
                                vc[l1] = _mm256_setzero_ps();
                            }

                            for (int k = 0; k < BM; k += 8) {
                                va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(a + i * BM + k)));
                                for (int l = 0; l < BLOCKB; l++) {
                                    vb[l] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(b + (j + l) * BM + k)));
                                }

                                for (int l1 = 0; l1 < BLOCKB; l1++) {
                                    vc[l1] = _mm256_fmadd_ps(va, vb[l1], vc[l1]);
                                }
                            }

                            __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
                            __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;

                            __t0 = _mm256_unpacklo_ps(vc[0], vc[1]);
                            __t1 = _mm256_unpackhi_ps(vc[0], vc[1]);
                            __t2 = _mm256_unpacklo_ps(vc[2], vc[3]);
                            __t3 = _mm256_unpackhi_ps(vc[2], vc[3]);
                            __t4 = _mm256_unpacklo_ps(vc[4], vc[5]);
                            __t5 = _mm256_unpackhi_ps(vc[4], vc[5]);
                            __t6 = _mm256_unpacklo_ps(vc[6], vc[7]);
                            __t7 = _mm256_unpackhi_ps(vc[6], vc[7]);
                            __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
                            __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
                            __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
                            __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
                            __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
                            __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
                            __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
                            __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));

                            __m256 sum = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
                            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt1, __tt5, 0x20));
                            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt2, __tt6, 0x20));
                            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt3, __tt7, 0x20));
                            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt0, __tt4, 0x31));
                            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt1, __tt5, 0x31));
                            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt2, __tt6, 0x31));
                            sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt3, __tt7, 0x31));

                            _mm_storeu_si128((__m128i *)(c + i * BK + j), _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
                        }
                    }

                    for (int i = nst; i < nend; i++) {
                        int j = kst;
                        for (; j + 7 < kend; j += 8) {
                            __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(outputData + i * k + j)));
                            __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(c + (i - nst) * BK + (j - kst))));
                            __m256 vsum = _mm256_add_ps(va, vb);
                            _mm_storeu_si128((__m128i *)(outputData + i * k + j), _mm256_cvtps_ph(vsum, _MM_FROUND_TO_NEAREST_INT));
                        }
                        for (; j < kend; j++) {
                            outputData[i * k + j] =
                                (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + fp16tofp32.dict[c[(i - nst) * BK + (j - kst)]]);
                        }
                    }
                }
            }
        }
        delete[] a;
        delete[] b;
        delete[] c;
    } else if (n > 0) {
        const int BLOCKA = 8;
        __m256 va[BLOCKA], vb, vc[BLOCKA];

        for (int i = 0; i < n; i += BLOCKA) {
            int cur = std::min(BLOCKA, n - i);
            for (int j = st; j < end; j++) {
                for (int l1 = 0; l1 < cur; l1++) {
                    vc[l1] = _mm256_setzero_ps();
                }

                for (int k = 0; k < m; k += 8) {
                    for (int l = 0; l < cur; l++) {
                        va[l] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(inputData + (i + l) * m + k)));
                    }
                    vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(weightData + j * m + k)));

                    for (int l1 = 0; l1 < cur; l1++) {
                        vc[l1] = _mm256_fmadd_ps(va[l1], vb, vc[l1]);
                    }
                }

                for (int l0 = 0; l0 < cur; l0++) {
                    float now = Floatsum(vc[l0]);
                    if (biasData != nullptr) {
                        now += biasData[j];
                    }
                    outputData[(i + l0) * k + j] = float_to_half(now);
                }
            }
        }
    }
#else
    if (n > 3) {
        int BN = 64, BM = 64, BK = 64;
        float *a = new float[BN * BM];
        float *b = new float[BK * BM];
        float *c = new float[BN * BK];

        if (biasData == nullptr) {
            for (int i = 0; i < n; i++) {
                memset(outputData + i * k + st, 0, (end - st) * sizeof(uint16_t));
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    outputData[i * k + j] = float_to_half(biasData[j]);
                }
            }
        }

        for (int mst = 0; mst < m; mst += BM) {
            memset(c, 0, BN * BK * sizeof(float));
            int mend = std::min(mst + BM, m);
            int mlen = mend - mst;
            for (int nst = 0; nst < n; nst += BN) {
                int nend = std::min(nst + BN, n);
                int nlen = nend - nst;
                for (int i = nst; i < nend; i++) {
                    Float16ToFloat32(inputData + i * m + mst, a + (i - nst) * BM, mend - mst);
                }

                for (int kst = st; kst < end; kst += BK) {
                    int kend = std::min(kst + BK, end);
                    int klen = kend - kst;
                    for (int i = kst; i < kend; i++) {
                        Float16ToFloat32(weightData + i * m + mst, b + (i - kst) * BM, mend - mst);
                    }

                    for (int i = 0; i < nlen; i++) {
                        float now = 0;
                        for (int j = 0; j < klen; j++) {
                            for (int g = 0; g < mlen; g++) {
                                now += a[i * BM + g] * b[j * BM + g];
                            }
                            c[i * BK + j] = now;
                        }
                    }

                    for (int i = nst; i < nend; i++) {
                        for (int j = kst; j < kend; j++) {
                            outputData[i * k + j] = (float_to_half)(g_bf16tofp32.dict[outputData[i * k + j]] + c[(i - nst) * BK + (j - kst)]);
                        }
                    }
                }
            }
        }
        delete[] a;
        delete[] b;
        delete[] c;
    } else {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
                for (; l < m; l++) {
                    now += g_bf16tofp32.dict[inputData[i * m + l]] * g_bf16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = float_to_half(now);
            }
        }
    }
#endif
}

void RunLinearFloat16Float16(uint16_t *inputData,
                             uint16_t *weightData,
                             uint16_t *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum) {
    int per = k / threadNum;
    int cur = 0;
    std::vector<MultiThreadLinearFloat16Float16Op *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = cur + per + (cur + per * (threadNum - i) < k);
        ops.push_back(new MultiThreadLinearFloat16Float16Op(inputData, weightData, biasData, outputData, n, m, k, cur, end));
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

// a = [n, m], b = [k, m], c = aT(b') = [n, k]
void RunLinearInt8Int8(uint8_t *a,
                       uint8_t *b,
                       float *c,
                       int n,
                       int m,
                       int k,
                       int *weightSums,
                       int *weightZeros,
                       float *scales,
                       float *bias,
                       float *inputSums,
                       float *iscales,
                       float *izeros,
                       AliveThreadPool *pool,
                       int startTid,
                       int threadNum) {
    int per = k / threadNum;
    int cur = 0;
    std::vector<MultiThreadLinearInt8Int8Op *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = cur + per + (cur + per * (threadNum - i) < k);
        if (i == threadNum - 1) {
            end = k;
        }
        ops.push_back(new MultiThreadLinearInt8Int8Op(a,
                                                      b + cur * m,
                                                      (int32_t *)c + cur,
                                                      n,
                                                      m,
                                                      end - cur,
                                                      k,
                                                      weightSums + cur,
                                                      weightZeros + cur,
                                                      scales + cur,
                                                      (bias == nullptr ? (float *)nullptr : bias + cur),
                                                      iscales,
                                                      izeros,
                                                      inputSums));
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

void RunLinearFloat32Int8(
    float *inputData, Data &weight, float *outputData, float *biasData, int n, int m, int k, AliveThreadPool *pool, int startTid, int threadNum) {
    weight.CalcWeightSum();
    std::vector<LowBitConfig> inputConfigs;
    std::vector<uint8_t> uinput;
    std::vector<float> inputSums, iscales, izeros;
    OnlineQuantization(inputData, uinput, inputConfigs, n, m, 1, m, inputSums, iscales, izeros, 0);

    RunLinearInt8Int8(uinput.data(),
                      (uint8_t *)weight.cpuData,
                      outputData,
                      n,
                      m,
                      k,
                      weight.weightSum.data(),
                      weight.zeros.data(),
                      weight.scales.data(),
                      biasData,
                      inputSums.data(),
                      iscales.data(),
                      izeros.data(),
                      pool,
                      startTid,
                      threadNum);
    /*
    这部分是float输入，float输出
    int threadNum = threads;
    int per = k / threadNum;
    int cur = 0;
    std::vector<std::thread *> threads;
    for (int i = 0; i < threadNum - 1; i++) {
        int end = cur + per + (cur + per * (threadNum - i) < k);
        threads.push_back(new std::thread(&Int8LinearPart, inputData, weightData, biasData, outputData,
                                            weight.perChannelsConfigs.data(), n, m, k, cur, end));
        cur = end;
    }
    Int8LinearPart(inputData, weightData, biasData, outputData, weight.perChannelsConfigs.data(), n, m, k, cur, k);
    for (int i = 0; i < threadNum - 1; i++) {
        threads[i]->join();
        delete threads[i];
    }
    */
}

void RunLinearFloat16Int4Group(uint16_t *inputData,
                               Data &weight,
                               uint16_t *outputData,
                               float *biasData,
                               int n,
                               int m,
                               int k,
                               int group,
                               int groupCnt,
                               AliveThreadPool *pool,
                               int startTid,
                               int threadNum) {
    std::vector<float> floatInput, floatOutput;
    floatInput.resize(n * m);
    floatOutput.resize(n * k);
    Float16ToFloat32(inputData, floatInput.data(), n * m);
    RunLinearFloat32Int4Group(floatInput.data(), weight, floatOutput.data(), biasData, n, m, k, group, groupCnt, pool, startTid, threadNum);
    Float32ToFloat16(floatOutput.data(), outputData, n * k);
}