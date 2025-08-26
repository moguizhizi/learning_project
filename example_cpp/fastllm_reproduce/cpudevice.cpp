#include "cpudevice.h"
#include "basellm.h"
#include "fastllm.h"
#include "file_utils.hpp"
#include "utils.h"
#include <cmath>
#include <cstring>

void CpuToFloat16::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end()) {
        ErrorInFastLLM("key error, key value is input");
    }

    Data &data = *(datas.find("input")->second);

    if (data.dims.size() == 0) {
        data.dataType == DataType::FLOAT16;
        data.UpdateUnitSize();
        return;
    }

    if (data.dataType == DataType::FLOAT16) {
        return;
    } else if (data.dataType == DataType::FLOAT32) {
        float *old = (float *)data.cpuData;
        int len = data.Count(0);
        data.dataType == DataType::FLOAT16;
        data.UpdateUnitSize();

        data.cpuData = new uint8_t[data.GetBytes()];
        uint16_t *cur = (uint16_t *)data.cpuData;

        for (int i = 0; i < len; i++) {
            cur[i] = float_to_half(old[i]);
        }
        delete[] old;
    } else {
        ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
    }
}

void CpuToFloat32::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end()) {
        ErrorInFastLLM("key error, key value is input");
    }

    Data &data = *(datas.find("input")->second);

    if (data.dims.size() == 0) {
        data.dataType == DataType::FLOAT32;
        data.UpdateUnitSize();
        return;
    }

    if (data.dataType == DataType::FLOAT32) {
        return;
    } else if (data.dataType == DataType::FLOAT16) {
        uint16_t *old = (uint16_t *)data.cpuData;
        int len = data.Count(0);
        data.dataType == DataType::FLOAT32;
        data.UpdateUnitSize();

        data.cpuData = new uint8_t[data.GetBytes()];
        float *cur = (float *)data.cpuData;

        for (int i = 0; i < len; i++) {
            cur[i] = g_fp16ToFp32Manager.dict[old[i]];
        }
        delete[] old;
    } else {
        ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
    }
}

void CpuConvertToFloat16::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is input or output");
    }

    Data *inputs = datas.find("input")->second;
    Data *outputs = datas.find("output")->second;
    outputs->dataType = DataType::FLOAT16;
    outputs->Resize(inputs->dims);
    if (inputs->expansionDims.size() > 0) {
        outputs->Expansion(inputs->expansionDims);
    }
}

void CpuConvertToFloat16::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is input or output");
    }

    Data *inputs = datas.find("input")->second;
    Data *outputs = datas.find("output")->second;
    outputs->Allocate();

    if (inputs->dataType == DataType::FLOAT16) {
        std::memcpy(outputs->cpuData, inputs->cpuData, inputs->GetBytes());
    } else if (inputs->dataType == DataType::FLOAT32) {
        Float32ToFloat16((float *)inputs->cpuData, (uint16_t *)outputs->cpuData, inputs->Count(0));
    } else {
        ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
    }
}

void CpuConvertToFloat32::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is input or output");
    }

    Data *inputs = datas.find("input")->second;
    Data *outputs = datas.find("output")->second;
    outputs->dataType = DataType::FLOAT32;
    outputs->Resize(inputs->dims);
    if (inputs->expansionDims.size() > 0) {
        outputs->Expansion(inputs->expansionDims);
    }
}

void CpuConvertToFloat32::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is input or output");
    }

    Data *inputs = datas.find("input")->second;
    Data *outputs = datas.find("output")->second;
    outputs->Allocate();

    if (inputs->dataType == DataType::FLOAT32) {
        std::memcpy(outputs->cpuData, inputs->cpuData, inputs->GetBytes());
    } else if (inputs->dataType == DataType::FLOAT16) {
        Float16ToFloat32((uint16_t *)inputs->cpuData, (float *)outputs->cpuData, inputs->Count(0));
    } else {
        ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
    }
}

void CpuAttention::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("q") == datas.end() || datas.find("k") == datas.end() || datas.find("v") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is q or k or v or output");
    }

    Data *q = datas.find("q")->second;
    Data *k = datas.find("k")->second;
    Data *v = datas.find("v")->second;
    Data *output = datas.find("output")->second;
    int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q->dims[0] / k->dims[0];

    AssertInFastLLM(q->dims.size() == 3 && k->dims.size() == 3 && v->dims.size() == 3, "Attention: dims of q, k, v should be 3.\n");
    AssertInFastLLM(q->dims[2] == k->dims[2], "Attention: q.dims[2] should be equal to k.dims[2].\n");
    AssertInFastLLM(k->dims[1] == v->dims[1], "Attention: k.dims[1] should be equal to v.dims[1].\n");
    AssertInFastLLM(k->dims[0] == v->dims[0], "Attention: k.dims[0] should be equal to v.dims[0].\n");
    AssertInFastLLM(q->dims[0] == k->dims[0] * group, "Attention: q.dims[0] should be equal to k.dims[0] * group.\n");

    AssertInFastLLM(q->dataType == k->dataType && q->dataType == v->dataType, "Attention: q, k, v's datatype should be same.\n");
    AssertInFastLLM(q->dataType == DataType::FLOAT32 || q->dataType == DataType::FLOAT16, "Attention's input's type should be float32.\n");

    std::vector<int> dims = {q->dims[0], q->dims[1], v->dims[2]};
    output->dataType = q->dataType;
    output->Resize(dims);
}

void CpuAttention::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("q") == datas.end() || datas.find("k") == datas.end() || datas.find("v") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is q or k or v or output");
    }

    Data &qd = *(datas.find("q")->second);
    Data &kd = *(datas.find("k")->second);
    Data &vd = *(datas.find("v")->second);
    Data &maskd = *(datas.find("mask")->second);
    Data &outputd = *(datas.find("output")->second);

    outputd.Allocate();

    int q0 = qd.dims[0], q1 = qd.dims[1], q2 = qd.dims[2];
    int k1 = kd.dims[1];
    int v2 = vd.dims[2];

    float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0f;
    int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : qd.dims[0] / kd.dims[0];

    uint64_t q_stride = qd.strides[0];
    uint64_t k_stride = kd.strides[0];
    uint64_t v_stride = vd.strides[0];
    uint64_t output_stride = outputd.strides[0];

    if (qd.dataType == DataType::FLOAT32) {
        float *q = (float *)qd.cpuData;
        float *k = (float *)kd.cpuData;
        float *v = (float *)vd.cpuData;
        float *output = (float *)outputd.cpuData;
        float *mask = maskd.dims.size() > 0 ? (float *)maskd.cpuData : nullptr;

        int batch = (mask != nullptr && maskd.dims.size() == 3) ? maskd.dims[0] : 1;
        batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;

        int maskStride = (mask != nullptr) ? (maskd.dims.size() == 3 ? maskd.strides[0] : maskd.Count(0)) : 0;
        std::fill(output, output + outputd.Count(0), 0.0f);

        // auto *pool = GetAlivePool();
        // int threads = pool->threads.size();

        std::vector<MultiThreadSingleAttentionOp *> ops;
        for (int o = 0; o < q0; o++) {
            ops.push_back(new MultiThreadSingleAttentionOp(q + o * q_stride,
                                                           k + (o / group) * k_stride,
                                                           v + (o / group) * v_stride,
                                                           mask + (o / (q0 / batch)) * maskStride,
                                                           output + o * output_stride,
                                                           scale,
                                                           q1,
                                                           q2,
                                                           k1,
                                                           v2));
        }

        // for (int st = 0; st < ops.size(); st += threads) {
        //     for (int i = st; i < ops.size() && i < st + threads; i++) {
        //         pool->PushOp(i - st, ops[i]);
        //     }
        //     for (int i = st; i < ops.size() && i < st + threads; i++) {
        //         pool->Wait(i - st);
        //     }
        // }
    } else if (qd.dataType == DataType::FLOAT16) {
        uint16_t *q = (uint16_t *)qd.cpuData;
        uint16_t *k = (uint16_t *)kd.cpuData;
        uint16_t *v = (uint16_t *)vd.cpuData;
        uint16_t *output = (uint16_t *)outputd.cpuData;
        uint16_t *mask = maskd.dims.size() > 0 ? (uint16_t *)maskd.cpuData : nullptr;

        int batch = (mask != nullptr && maskd.dims.size() == 3) ? maskd.dims[0] : 1;
        batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;

        int maskStride = (mask != nullptr) ? (maskd.dims.size() == 3 ? maskd.strides[0] : maskd.Count(0)) : 0;
        std::fill(output, output + outputd.Count(0), 0.0f);

        // auto *pool = GetAlivePool();
        // int threads = pool->threads.size();

        std::vector<MultiThreadSingleAttentionFloat16Op *> ops;
        for (int o = 0; o < q0; o++) {
            ops.push_back(new MultiThreadSingleAttentionFloat16Op(q + o * q_stride,
                                                                  k + (o / group) * k_stride,
                                                                  v + (o / group) * v_stride,
                                                                  mask + (o / (q0 / batch)) * maskStride,
                                                                  output + o * output_stride,
                                                                  scale,
                                                                  q1,
                                                                  q2,
                                                                  k1,
                                                                  v2));
        }

        // for (int st = 0; st < ops.size(); st += threads) {
        //     for (int i = st; i < ops.size() && i < st + threads; i++) {
        //         pool->PushOp(i - st, ops[i]);
        //     }
        //     for (int i = st; i < ops.size() && i < st + threads; i++) {
        //         pool->Wait(i - st);
        //     }
        // }
    } else {
        ErrorInFastLLM("Attention error: unsupport dataType.\n");
    }
}

void CpuCopyKVCacheOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) { return; }

void CpuCopyKVCacheOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("newCache") == datas.end() || datas.find("oldCache") == datas.end()) {
        ErrorInFastLLM("key error, key value is newKVCache or oldKVCache");
    }

    Data &newCache = *(datas.find("newCache")->second);
    Data &oldCache = *(datas.find("oldCache")->second);
    int newBsStart = intParams.find("newBsStart") != intParams.end() ? intParams.find("newBsStart")->second : 0;
    int oldBsStart = intParams.find("oldBsStart") != intParams.end() ? intParams.find("oldBsStart")->second : 0;
    int bs = intParams.find("bs") != intParams.end() ? intParams.find("bs")->second : 0;
    int offset = intParams.find("offset") != intParams.end() ? intParams.find("offset")->second : 0;

    int unitSize = oldCache.unitSize;
    for (int o = 0; o < bs; o++) {
        uint8_t *cur = newCache.cpuData + (newBsStart + o) * newCache.strides[0] * unitSize;
        cur = cur + offset * newCache.strides[1] * unitSize;
        uint8_t *old = oldCache.cpuData + (oldBsStart + o) * oldCache.strides[0] * unitSize;
        std::memcpy(cur, old, oldCache.dims[1] * oldCache.dims[2] * unitSize);
    }
}

void CpuEmbedding::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("weight") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is input or weight or output");
    }

    Data &input = *(datas.find("input")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &output = *(datas.find("output")->second);

    AssertInFastLLM(weight.dims.size() == 2, "Embedding's weight's dim should be 2.\n");
    AssertInFastLLM(weight.dataType == DataType::FLOAT32 || weight.dataType == DataType::FLOAT16 || weight.dataType == DataType::BFLOAT16,
                    "Embedding's weight's type should be float32 or float16 or bfloat16.\n");
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Embedding's input's type should be float32 or float16.\n");

    weight.weightType = WeightType::EMBEDDING;
    int vocabSize = weight.dims[0];
    int embeddingsSize = weight.dims[1];

    std::vector<int> dims = input.dims;
    dims.push_back(embeddingsSize);
    output.dataType = input.dataType;
    if (weight.dataType == DataType::FLOAT16) {
        output.dataType = DataType::FLOAT16;
    }

    output.Resize(dims);
}

void CpuEmbedding::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("weight") == datas.end() || datas.find("output") == datas.end()) {
        ErrorInFastLLM("key error, key value is input or weight or output");
    }

    Data &input = *(datas.find("input")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &output = *(datas.find("output")->second);

    output.Allocate();

    int inputlen = input.dims[0];
    int embeddingsSize = weight.dims[1];

    std::vector<float> tempInputData;
    std::vector<float> tempOutputData;

    tempInputData.resize(inputlen);
    tempOutputData.resize(inputlen * embeddingsSize);

    float *inputData = tempInputData.data();
    float *outputData = tempOutputData.data();

    for (int i = 0; i < inputlen; i++) {
        if (input.dataType == DataType::FLOAT32) {
            inputData[i] = ((float *)input.cpuData)[i];
        } else if (input.dataType == DataType::FLOAT16) {
            inputData[i] = half_to_float(((uint16_t *)input.cpuData)[i]);
        } else {
            ErrorInFastLLM("Embedding error: unsupport dataType.\n");
        }
    }

    if (GetLowMemMode()) {
        FILE *file = fopen(weight.fileName.c_str(), "rb");
        if (weight.dataType == DataType::FLOAT32) {
            for (int i = 0; i < inputlen; i++) {
                int inputId = (int)(inputData[i] + 1e-9);
#if defined(_WIN32) or defined(_WIN64)
                _fseeki64(file, weight.filePos + (long long)inputId * embeddingsSize * sizeof(float), 0);
#else
                fseek(file, weight.filePos + (long long)inputId * embeddingsSize * sizeof(float), 0);
#endif
                fread(outputData + i * embeddingsSize, sizeof(float), embeddingsSize, file);
            }
        } else if (weight.dataType == DataType::FLOAT16) {
            for (int i = 0; i < inputlen; i++) {
                int inputId = (int)(inputData[i] + 1e-9);
#if defined(_WIN32) or defined(_WIN64)
                _fseeki64(file, weight.filePos + (long long)inputId * embeddingsSize * sizeof(uint16_t), 0);
#else
                fseek(file, weight.filePos + (long long)inputId * embeddingsSize * sizeof(uint16_t), 0);
#endif

                uint16_t *temp = new uint16_t[embeddingsSize];
                std::memset(temp, 0, embeddingsSize * sizeof(uint16_t));
                fread(temp, sizeof(uint16_t), embeddingsSize, file);

                for (int j = 0; j < embeddingsSize; j++) {
                    outputData[i * embeddingsSize + j] = half_to_float(temp[j]);
                }
                delete[] temp;
            }
        } else {
        }
        fclose(file);
    } else {
        if (weight.dataType == DataType::FLOAT32) {
            for (int i = 0; i < inputlen; i++) {
                int inputId = (int)(inputData[i] + 1e-9);
                std::memcpy(outputData + i * embeddingsSize * sizeof(float),
                            weight.cpuData + inputId * embeddingsSize * sizeof(float),
                            embeddingsSize * sizeof(float));
            }
        } else if (weight.dataType == DataType::FLOAT16) {
            for (int i = 0; i < inputlen; i++) {
                int inputId = (int)(inputData[i] + 1e-9);
                for (int j = 0; j < embeddingsSize; j++) {
                    outputData[i * embeddingsSize + j] = half_to_float(((uint16_t *)weight.cpuData)[inputId * embeddingsSize + j]);
                }
            }
        }
    }

    if (output.dataType == DataType::FLOAT32) {
        std::memcpy(output.cpuData, (uint8_t *)outputData, inputlen * embeddingsSize * sizeof(float));
    } else if (output.dataType == DataType::FLOAT16) {
        for (int i = 0; i < inputlen * embeddingsSize; i++) {
            ((uint16_t *)output.cpuData)[i] = float_to_half(outputData[i]);
        }
    } else {
        ErrorInFastLLM("Embedding error: unsupport dataType.\n");
    }
}

void CpuLayerNormOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("output") == datas.end() || datas.find("gamma") == datas.end() ||
        datas.find("beta") == datas.end()) {

        ErrorInFastLLM("key error, key value is input or output or gamma or beta");
    }

    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &gamma = *(datas.find("gamma")->second);
    Data &beta = *(datas.find("beta")->second);

    output.Allocate();

    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.strides[axis];

    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;
    float *gammaData = (float *)gamma.cpuData;
    float *betaData = (float *)beta.cpuData;

    float *inputWalk = nullptr;
    float *outputWalk = nullptr;

    if (inner == 1) {

        for (int i = 0; i < outer; i++) {
            float mean = 0.f, s2 = 0.f, var = 0.f;
            int j = 0;
#ifdef __aarch64__
            float32x4_t sums = vdupq_n_f32(0.0);
            float32x4_t sums2 = vdupq_n_f32(0.0);
            for (; j + 3 < channels; j += 4) {
                float32x4_t vi = vld1q_f32(inputData + j);
                sums = vaddq_f32(sums, vi);
                sums2 = vaddq_f32(sums2, vmulq_f32(vi, vi));
            }
            mean = sums[0] + sums[1] + sums[2] + sums[3];
            s2 = sums2[0] + sums2[1] + sums2[2] + sums2[3];
#endif
#ifdef __AVX2__
            __m256 sum_vec = _mm256_setzero_ps();
            __m256 squared_sum_vec = _mm256_setzero_ps();

            for (; j < channels - 7; j += 8) {
                __m256 data_vec = _mm256_loadu_ps(inputData + j);
                sum_vec = _mm256_add_ps(sum_vec, data_vec);

                __m256 squared_data_vec = _mm256_mul_ps(data_vec, data_vec);
                squared_sum_vec = _mm256_add_ps(squared_sum_vec, squared_data_vec);
            }

            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            mean = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

            float squared_sum_array[8];
            _mm256_storeu_ps(squared_sum_array, squared_sum_vec);
            s2 = squared_sum_array[0] + squared_sum_array[1] + squared_sum_array[2] + squared_sum_array[3] + squared_sum_array[4] +
                 squared_sum_array[5] + squared_sum_array[6] + squared_sum_array[7];
#endif
            for (; j < channels; j++) {
                mean += inputData[j];
                s2 += inputData[j] * inputData[j];
            }
            mean /= channels;
            var = sqrt(s2 / channels - mean * mean + 1e-10);
            j = 0;
#ifdef __aarch64__
            float32x4_t means = vdupq_n_f32(mean);
            float32x4_t vars = vdupq_n_f32(1.0 / var);
            for (; j + 3 < channels; j += 4) {
                float32x4_t va = vld1q_f32(gammaData + j), vb = vld1q_f32(betaData + j);
                float32x4_t vi = vld1q_f32(inputData + j);
                float32x4_t vo = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(vi, means), vars), va), vb);
                vst1q_f32(outputData + j, vo);
            }
#endif
            for (; j < channels; j++) {
                float a = gammaData[j], b = betaData[j];
                outputData[j] = (inputData[j] - mean) / var * a + b;
            }

            inputData += channels;
            outputData += channels;
        }
        return;

    } else {

        float *mean = new float[inner]();
        float *var = new float[inner]();

        for (int i = 0; i < outer; i++) {
            std::fill(mean, mean + inner, 0.0f);
            std::fill(var, var + inner, 0.0f);

            inputWalk = inputData;
            outputWalk = outputData;
            for (int j = 0; j < channels; j++) {
                for (int k = 0; k < inner; k++) {
                    mean[k] = mean[k] + *inputWalk;
                    inputWalk = inputWalk + 1;
                }
            }

            for (int k = 0; k < inner; k++) {
                mean[k] = mean[k] / channels;
            }

            inputWalk = inputData;
            for (int j = 0; j < channels; j++) {
                for (int k = 0; k < inner; k++) {
                    float x = *inputWalk - mean[k];
                    var[k] = var[k] + x * x;
                    inputWalk = inputWalk + 1;
                }
            }

            for (int k = 0; k < inner; k++) {
                var[k] = std::sqrt(var[k] / channels + 1e-5);
            }

            inputWalk = inputData;
            for (int j = 0; j < channels; j++) {
                float a = gammaData[j];
                float b = betaData[j];
                for (int k = 0; k < inner; k++) {
                    *outputWalk = (*inputWalk - mean[k]) / var[k] * a + b;
                    inputWalk = inputWalk + 1;
                    outputWalk = outputWalk + 1;
                }
            }

            inputData = inputData + channels * inner;
            outputData = outputData + channels * inner;
        }

        delete[] mean;
        delete[] var;
    }
}

void RunMultiThreadRMSNormFloat(float *output, float *input, float *weight, int outer, int channels, float eps, AliveThreadPool *pool) {
    if (outer == 1) {
        (MultiThreadRMSNormFloatOp(output, input, weight, outer, channels, eps)).Run();
        return;
    }
    int threadNum = pool->threads.size();
    int per = outer / pool->threads.size();
    int cur = 0;
    std::vector<MultiThreadRMSNormFloatOp *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
        ops.push_back(new MultiThreadRMSNormFloatOp(output + cur * channels, input + cur * channels, weight, end - cur, channels, eps));
        cur = end;
    }
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(i, ops[i]);
    }
    for (int i = 0; i < threadNum; i++) {
        pool->Wait(i);
        delete ops[i];
    }
}

void CpuRMSNormOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();

    float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (input.dataType == DataType::FLOAT32) {
        float *inputData = (float *)input.cpuData;
        float *outputData = (float *)output.cpuData;
        float *weightData = (float *)weight.cpuData;
        RunMultiThreadRMSNormFloat(outputData, inputData, weightData, outer, channels, eps, GetAlivePool());
    } else if (input.dataType == DataType::FLOAT16) {
        uint16_t *inputData = (uint16_t *)input.cpuData;
        uint16_t *outputData = (uint16_t *)output.cpuData;
        float *weightData = (float *)weight.cpuData;

        for (int i = 0; i < outer; i++) {
            float mean = 0.f;
            int j = 0;
            for (; j < channels; j++) {
                float x = g_fp16ToFp32Manager.dict[inputData[j]];
                mean += x * x;
            }
            float scale = 1.0 / sqrt(mean / channels + eps);
            j = 0;
            for (; j < channels; j++) {
                outputData[j] = float_to_half(g_fp16ToFp32Manager.dict[inputData[j]] * scale * weightData[j]);
            }

            inputData += channels;
            outputData += channels;
        }
    } else {
        ErrorInFastLLM("RMSNorm error: unsupport dataType.\n");
    }
}

bool CpuConv2DOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) { return true; }

void CpuConv2DOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);

    int kernelH = intParams.find("kernelH")->second;
    int kernelW = intParams.find("kernelW")->second;
    int padH = intParams.find("padH")->second;
    int padW = intParams.find("padW")->second;
    int strideH = intParams.find("strideH")->second;
    int strideW = intParams.find("strideW")->second;
    int outputChannels = intParams.find("outputChannels")->second;
    int inputChannels = intParams.find("inputChannels")->second;

    AssertInFastLLM(weight.dims.size() == 4, "Conv2D's weight's shape's size should be 4.\n");
    AssertInFastLLM(input.dims[1] == inputChannels, "Conv2D's input's shape error.\n");

    int inputHeight = input.dims[2];
    int inputWidth = input.dims[3];

    int outputHeight = (inputHeight + 2 * padH - kernelH) / strideH + 1;
    int outputWidth = (inputWidth + 2 * padW - kernelW) / strideW + 1;

    weight.weightType = WeightType::CONV2D;

    std::vector<int> dims = input.dims;
    dims[1] = outputChannels;
    dims[2] = outputHeight;
    dims[3] = outputWidth;

    output.dataType = input.dataType;
    output.Resize(dims);
}

void CpuConv2DOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &bias = *(datas.find("bias")->second);

    output.Allocate(0.0f);

    int kernelH = intParams.find("kernelH")->second;
    int kernelW = intParams.find("kernelW")->second;
    int padH = intParams.find("padH")->second;
    int padW = intParams.find("padW")->second;
    int strideH = intParams.find("strideH")->second;
    int strideW = intParams.find("strideW")->second;
    int outputChannels = intParams.find("outputChannels")->second;
    int inputChannels = intParams.find("inputChannels")->second;

    int inputHeight = input.dims[2];
    int inputWidth = input.dims[3];
    int outputHeight = (inputHeight + 2 * padH - kernelH) / strideH + 1;
    int outputWidth = (inputWidth + 2 * padW - kernelW) / strideW + 1;

    float *floatInput = (float *)input.cpuData;
    float *floatOutput = (float *)output.cpuData;
    float *floatWeight = (float *)weight.cpuData;
    float *floatBias = (float *)bias.cpuData;

    for (int oc = 0; oc < outputChannels; oc++) {
        float *startWeight = floatWeight + oc * (inputChannels * kernelH * kernelW);
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                int ih = oh * kernelH - padH;
                int iw = ow * strideW - padW;
                float value = floatBias[oc];

                float *curWeight = startWeight;
                for (int c = 0; c < inputChannels; c++) {
                    float *curInput = floatInput + c * (inputHeight * inputWidth);
                    for (int h = 0; h < kernelH; h++) {
                        for (int w = 0; w < kernelW; w++) {
                            int x = ih + h;
                            int y = iw + w;
                            if (x >= 0 && x <= inputHeight && y >= 0 && y <= inputWidth) {
                                value = value + curInput[x * inputWidth + y] * (*curWeight);
                                curWeight++;
                            }
                        }
                    }
                }
                *floatOutput = value;
                floatOutput++;
            }
        }
    }
}

void DoCpuLinearReshape(Data &input, Data &weight, Data &output) {
    weight.weightType = WeightType::LINEAR;
    std::vector<int> dims = input.dims;
    dims.back() = weight.dims[0];

    output.dataType = input.dataType;
    output.Resize(dims);
}

void CpuLinearOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);

    AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
    AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

    DoCpuLinearReshape(input, weight, output);
}

// float的input, int8的weight, 直接计算得到float的output
void Int8LinearPart(
    float *inputData, uint8_t *weightData, float *biasData, float *outputData, LowBitConfig *configs, int n, int m, int k, int st, int end) {
    for (int i = 0; i < n; i++) {
        for (int j = st; j < end; j++) {
            float now = biasData ? biasData[j] : 0.0f;
            int l = 0;

#ifdef __aarch64__
            float32x4_t scales = vdupq_n_f32(configs[j].scale);
            uint8x8_t zeros = vdup_n_u8(configs[j].zeroPoint);
            float32x4_t sum0 = {0, 0, 0, 0};
            float32x4_t sum1 = {0, 0, 0, 0};
            for (; l + 7 < m; l += 8) {
                uint8x8_t a = vld1_u8(weightData + j * m + l);
                uint16x8_t result = vsubl_u8(a, zeros);
                int16x8_t sresult = vreinterpretq_s16_u16(result);
                int16x4_t result1 = vget_low_s16(sresult);
                int16x4_t result2 = vget_high_s16(sresult);
                int32x4_t result3 = vmovl_s16(result1);
                int32x4_t result4 = vmovl_s16(result2);
                float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                sum0 = vaddq_f32(sum0, vmulq_f32(vld1q_f32(inputData + i * m + l + 0), f1));
                sum1 = vaddq_f32(sum1, vmulq_f32(vld1q_f32(inputData + i * m + l + 4), f2));
            }
            now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
            now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif

            for (; l < m; l++) {
                now = now + inputData[i * m + l] * configs[j].invQuantization(weightData[j * m + l]);
            }

            outputData[i * k + j] = now;
        }
    }
}

// float的input, int4g的weight, 直接计算得到float的output
void Int4GroupLinearPart(float *inputData,
                         uint8_t *weightData,
                         float *biasData,
                         float *outputData,
                         LowBitConfig *configs,
                         int n,
                         int m,
                         int k,
                         int st,
                         int end,
                         int group,
                         int groupCnt) {
    for (int i = 0; i < n; i++) {
        for (int j = st; j < end; j++) {
            float now = biasData ? biasData[j] : 0.0f;

            for (int g = 0; g < group; g++) {
                int gst = g * groupCnt;
                int gend = std::min((g + 1) * groupCnt, m);
                int l = gst;
#ifdef __aarch64__
                float32x4_t scales = vdupq_n_f32(configs[j * group + g].scale);
                uint8x8_t zeros = vdup_n_u8(configs[j * group + g].zeroPoint);
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                float32x4_t sum0 = {0, 0, 0, 0};
                float32x4_t sum1 = {0, 0, 0, 0};

                for (; l + 15 < gend; l += 16) {
                    uint8x8_t ori = vld1_u8(weightData + (j * m + l) / 2);
                    float32x4x2_t in0 = vld2q_f32(inputData + i * m + l + 0);
                    float32x4x2_t in1 = vld2q_f32(inputData + i * m + l + 8);
                    uint8x8_t a = vand_u8(ori, maskLow);
                    uint16x8_t result = vsubl_u8(a, zeros);
                    int16x8_t sresult = vreinterpretq_s16_u16(result);
                    int16x4_t result1 = vget_low_s16(sresult);
                    int16x4_t result2 = vget_high_s16(sresult);
                    int32x4_t result3 = vmovl_s16(result1);
                    int32x4_t result4 = vmovl_s16(result2);
                    float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));
                    sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[1], f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[1], f2));

                    a = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                    result = vsubl_u8(a, zeros);
                    sresult = vreinterpretq_s16_u16(result);
                    result1 = vget_low_s16(sresult);
                    result2 = vget_high_s16(sresult);
                    result3 = vmovl_s16(result1);
                    result4 = vmovl_s16(result2);
                    f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                    sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[0], f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[0], f2));
                }
                now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif
                for (; l < gend; l++) {
                    int id = (j * m + l) / 2;
                    float weight = 0.0f;
                    if ((j * m + l) % 2) {
                        weight = configs[j * group + g].invQuantization(weightData[id] & 0xF);
                    } else {
                        weight = configs[j * group + g].invQuantization(weightData[id] >> 4);
                    }
                    now += inputData[i * m + l] * weight;
                }
            }

            outputData[i * k + j] = now;
        }
    }
}