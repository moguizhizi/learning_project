#include "cpudevice.h"
#include "basellm.h"
#include "computeutils.h"
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

        auto *pool = GetAlivePool();
        int threads = pool->threads.size();

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

        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
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

        auto *pool = GetAlivePool();
        int threads = pool->threads.size();

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

        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
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

void DoCpuLinear(Data &input, Data &weight, const Data &bias, Data &output) {
    // auto st = std::chrono::system_clock::now();
    output.Allocate();
    int n = input.Count(0) / input.dims.back();
    int m = input.dims.back();
    int k = output.dims.back();
    int threadSt = GetAlivePool()->curActivateThreadInterval.first;
    int threadLen = GetAlivePool()->curActivateThreadInterval.second - GetAlivePool()->curActivateThreadInterval.first;

    if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32) {
        if (weight.dataType == DataType::FLOAT32) {
            RunLinearFloat32Float32((float *)input.cpuData,
                                    (float *)weight.cpuData,
                                    (float *)output.cpuData,
                                    bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                    n,
                                    m,
                                    k,
                                    GetAlivePool(),
                                    threadSt,
                                    threadLen);
        } else if (weight.dataType == DataType::FLOAT16) {
            RunLinearFloat32Float16((float *)input.cpuData,
                                    (uint16_t *)weight.cpuData,
                                    (float *)output.cpuData,
                                    bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                    n,
                                    m,
                                    k,
                                    GetAlivePool(),
                                    threadSt,
                                    threadLen);
        } else if (weight.dataType == DataType::INT8) {
            RunLinearFloat32Int8((float *)input.cpuData,
                                 weight,
                                 (float *)output.cpuData,
                                 bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                 n,
                                 m,
                                 k,
                                 GetAlivePool(),
                                 threadSt,
                                 threadLen);
        } else if (weight.dataType == DataType::INT4_GROUP || weight.dataType == DataType::INT4_NOZERO) {
            int group = weight.group, groupCnt = weight.groupCnt;
            if (weight.dataType == DataType::INT4_NOZERO) {
                group = 1, groupCnt = m;
            }
            RunLinearFloat32Int4Group((float *)input.cpuData,
                                      weight,
                                      (float *)output.cpuData,
                                      bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                      n,
                                      m,
                                      k,
                                      group,
                                      groupCnt,
                                      GetAlivePool(),
                                      threadSt,
                                      threadLen);
        } else if (weight.dataType == DataType::INT2_GROUP) {
            int group = weight.group, groupCnt = weight.groupCnt;
            RunLinearFloat32Int2Group((float *)input.cpuData,
                                      weight,
                                      (float *)output.cpuData,
                                      bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                      n,
                                      m,
                                      k,
                                      group,
                                      groupCnt,
                                      GetAlivePool(),
                                      threadSt,
                                      threadLen);
        } else if (weight.dataType == DataType::BASE3_GROUP) {
            std::vector<uint8_t> base = {1, 3, 9, 27, 81};
            float *inputData = (float *)input.cpuData;
            uint8_t *weightData = (uint8_t *)weight.cpuData;
            float *outputData = (float *)output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr;

            auto pool = GetAlivePool();
            int threadNum = pool->threads.size();
            int per = k / threadNum;
            int cur = 0;
            std::vector<MultiThreadBase3GroupLinearOp *> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                if (i == threadNum - 1) {
                    end = k;
                }
                ops.push_back(new MultiThreadBase3GroupLinearOp(
                    inputData, weightData, biasData, outputData, n, m, k, cur, end, weight.group, weight.groupCnt, weight.halfScales.data()));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        } else if (weight.dataType == DataType::INT4) {
            // 目前已经不用这种数据类型了
            float *inputData = (float *)input.cpuData;
            uint8_t *weightData = (uint8_t *)weight.cpuData;
            float *outputData = (float *)output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr;
            weight.CalcWeightSum();

            std::vector<LowBitConfig> inputConfigs;
            std::vector<uint8_t> uinput;
            std::vector<float> inputSums, iscales, izeros;
            OnlineQuantization(inputData, uinput, inputConfigs, n, m, 1, m, inputSums, iscales, izeros, 1);
            MultiplyInt4MultiThread(uinput.data(),
                                    weightData,
                                    (int32_t *)outputData,
                                    n,
                                    m,
                                    k,
                                    weight.weightSum.data(),
                                    weight.zeros.data(),
                                    weight.scales.data(),
                                    biasData,
                                    inputConfigs,
                                    GetThreads());
        } else if (weight.dataType == DataType::FP8_E4M3) {
            RunLinearFloat32FP8E4M3((float *)input.cpuData,
                                    weight,
                                    (float *)output.cpuData,
                                    bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                    n,
                                    m,
                                    k,
                                    GetAlivePool(),
                                    threadSt,
                                    threadLen);
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
    } else if (input.dataType == DataType::FLOAT16 && output.dataType == DataType::FLOAT16) {
        if (weight.dataType == DataType::FLOAT32) {
            RunLinearFloat16Float32((uint16_t *)input.cpuData,
                                    (float *)weight.cpuData,
                                    (uint16_t *)output.cpuData,
                                    bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                    n,
                                    m,
                                    k,
                                    GetAlivePool(),
                                    threadSt,
                                    threadLen);
        } else if (weight.dataType == DataType::FLOAT16) {
            RunLinearFloat16Float16((uint16_t *)input.cpuData,
                                    (uint16_t *)weight.cpuData,
                                    (uint16_t *)output.cpuData,
                                    bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                    n,
                                    m,
                                    k,
                                    GetAlivePool(),
                                    threadSt,
                                    threadLen);
        } else if (weight.dataType == DataType::INT8) {
            RunLinearFloat16Int8((uint16_t *)input.cpuData,
                                 weight,
                                 (uint16_t *)output.cpuData,
                                 bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                 n,
                                 m,
                                 k,
                                 GetAlivePool(),
                                 threadSt,
                                 threadLen);
        } else if (weight.dataType == DataType::INT4_GROUP || weight.dataType == DataType::INT4_NOZERO) {
            int group = weight.group, groupCnt = weight.groupCnt;
            if (weight.dataType == DataType::INT4_NOZERO) {
                group = 1, groupCnt = m;
            }
            RunLinearFloat16Int4Group((uint16_t *)input.cpuData,
                                      weight,
                                      (uint16_t *)output.cpuData,
                                      bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                      n,
                                      m,
                                      k,
                                      group,
                                      groupCnt,
                                      GetAlivePool(),
                                      threadSt,
                                      threadLen);
        } else if (weight.dataType == DataType::FP8_E4M3) {
            RunLinearFloat16FP8E4M3((uint16_t *)input.cpuData,
                                    weight,
                                    (uint16_t *)output.cpuData,
                                    bias.dims.size() > 0 ? (float *)bias.cpuData : nullptr,
                                    n,
                                    m,
                                    k,
                                    GetAlivePool(),
                                    threadSt,
                                    threadLen);
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
    } else {
        ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
    }
    // float spend = GetSpan(st, std::chrono::system_clock::now());
    // float gops = (float)n * m * k / spend / 1e9;
    //  printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
}

void CpuLinearOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);

    AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
    AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

    DoCpuLinearReshape(input, weight, output);
}

bool CpuLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (intParams.find("exType") != intParams.end()) {
        return false;
    }
    return true;
}

void CpuLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &bias = *(datas.find("bias")->second);
    AssertInFastLLM(bias.dataType == DataType::FLOAT32, "Linear's bias' type should be float32.\n");
    DoCpuLinear(input, weight, bias, output);
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

// float的input, int4的weight, 直接计算得到float的output
void Int4LinearPart(
    float *inputData, uint8_t *weightData, float *biasData, float *outputData, LowBitConfig *configs, int n, int m, int k, int st, int end) {
    for (int i = 0; i < n; i++) {
        for (int j = st; j < end; j++) {
            float now = biasData ? biasData[j] : 0.0f;
            int l = 0;
#ifdef __aarch64__X
            float32x4_t scales = vdupq_n_f32(configs[j].scale);
            uint8x8_t zeros = vdup_n_u8(configs[j].zeroPoint);
            uint8x8_t maskHigh = vdup_n_u8(0xF0);
            uint8x8_t maskLow = vdup_n_u8(0xF);
            float32x4_t sum0 = {0, 0, 0, 0};
            float32x4_t sum1 = {0, 0, 0, 0};

            for (; l + 15 < m; l += 16) {
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

            for (; l < m; l++) {
                int id = (j * m + l) / 2;
                float weight = 0.0f;
                if ((j * m + l) % 2) {
                    weight = configs[j].invQuantization(weightData[id] & 0xF);
                } else {
                    weight = configs[j].invQuantization(weightData[id] >> 4);
                }
                now += inputData[i * m + l] * weight;
            }

            outputData[i * k + j] = now;
        }
    }
}

MultiThreadLinearInt4Op::MultiThreadLinearInt4Op(uint8_t *a,
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
                                                 LowBitConfig *config,
                                                 int *inputSums) {

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
    this->config = config;
    this->inputSums = inputSums;
}

void MultiThreadLinearInt4Op::Run() {
    int block = 0;
    for (; block < n; block++) {
        uint32_t inputSum = inputSums[block];
        uint8_t *weightWalk = b;
        uint8_t *inputStart = a + block * m;

        for (int i = 0; i < k; i++) {
            int value = 0;
            uint8_t *inputWalk = inputStart;
            int j = 0;
#ifdef __ARM_FEATURE_DOTPROD
            uint8x8_t maskHigh = vdup_n_u8(0xF0);
            uint8x8_t maskLow = vdup_n_u8(0xF);
            uint32x2_t sum0 = {0, 0};

            for (; j + 15 < m; j += 16) {
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

            for (; j + 15 < m; j += 16) {
                uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                uint8x8x2_t in = vld2_u8(inputWalk + j);
                uint8x8_t va = vand_u8(ori, maskLow);
                uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
            }
            value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
            value += DotU4U8(weightWalk + i * m / 2, inputWalk, m);
            j += m;
#endif
            for (; j + 1 < m; j += 2) {
                int id = (i * m + j) / 2;
                value += (weightWalk[id] >> 4) * inputWalk[j];
                value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
            }

            for (; j < m; j++) {
                int id = (i * m + j) / 2;
                if ((i * m + j) % 2) {
                    value += (weightWalk[id] & 0xF) * inputWalk[j];
                } else {
                    value += (weightWalk[id] >> 4) * inputWalk[j];
                }
            }

            value -= weightSums[i] * config[block].zeroPoint;
            value -= inputSum * weightZeros[i];
            value += (int)config[block].zeroPoint * weightZeros[i] * m;

            ((float *)c)[block * kstride + i] = scales[i] * config[block].scale * value + (bias == nullptr ? 0.0 : bias[i]);
        }
    }
}

// a = [n, m], b = [k, m], c = aT(b') = [n, k]
void MultiplyInt4MultiThread(uint8_t *a,
                             uint8_t *b,
                             int32_t *c,
                             int n,
                             int m,
                             int k,
                             int *weightSums,
                             int *weightZeros,
                             float *scales,
                             float *bias,
                             std::vector<LowBitConfig> &configs,
                             int threadNum) {
    std::vector<int> inputSums;
    for (int i = 0; i < n; i++) {
        int sum = 0;
        for (int j = 0; j < m; j++) {
            sum += a[i * m + j];
        }
        inputSums.push_back(sum);
    }
    auto *pool = GetAlivePool();
    threadNum = pool->threads.size();
    int per = k / threadNum;
    int cur = 0;
    if (threadNum == 1) {
        MultiThreadLinearInt4Op(a,
                                b + cur * m / 2,
                                c + cur,
                                n,
                                m,
                                k - cur,
                                k,
                                weightSums + cur,
                                weightZeros + cur,
                                scales + cur,
                                (bias == nullptr ? (float *)nullptr : bias + cur),
                                configs.data(),
                                inputSums.data())
            .Run();
    } else {
        std::vector<MultiThreadLinearInt4Op *> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
            ops.push_back(new MultiThreadLinearInt4Op(a,
                                                      b + cur * m / 2,
                                                      c + cur,
                                                      n,
                                                      m,
                                                      end - cur,
                                                      k,
                                                      weightSums + cur,
                                                      weightZeros + cur,
                                                      scales + cur,
                                                      (bias == nullptr ? (float *)nullptr : bias + cur),
                                                      configs.data(),
                                                      inputSums.data()));
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
}

void GetArrayMinMax(float *a, int len, float &minValue, float &maxValue) {
    int j = 0;
    minValue = 1e100;
    maxValue = -1e100;
#ifdef __aarch64__
    float32x4_t mins = vdupq_n_f32(1e100);
    float32x4_t maxs = vdupq_n_f32(-1e100);
    for (; j + 3 < len; j += 4) {
        float32x4_t v = vld1q_f32(a + j);
        mins = vminq_f32(mins, v);
        maxs = vmaxq_f32(maxs, v);
    }
    for (int l = 0; l < 4; l++) {
        minValue = std::min(minValue, mins[l]);
        maxValue = std::max(maxValue, maxs[l]);
    }
#endif
#ifdef __AVX2__
    __m256 mins = _mm256_set1_ps(1e100);
    __m256 maxs = _mm256_set1_ps(-1e100);
    for (; j + 7 < len; j += 8) {
        __m256 v = _mm256_loadu_ps(a + j);
        mins = _mm256_min_ps(mins, v);
        maxs = _mm256_max_ps(maxs, v);
    }
    // 将 AVX2 寄存器中的最小值、最大值提取到标量
    float temp_min[8], temp_max[8];
    _mm256_storeu_ps(temp_min, mins);
    _mm256_storeu_ps(temp_max, maxs);
    for (int l = 0; l < 8; l++) {
        minValue = std::min(minValue, temp_min[l]);
        maxValue = std::max(maxValue, temp_max[l]);
    }
#endif
    for (; j < len; j++) {
        minValue = std::min(minValue, a[j]);
        maxValue = std::max(maxValue, a[j]);
    }
}

void QuantizationAll(float *fValue, uint8_t *uValue, int len, LowBitConfig *config) {
    float scale = config->scale;
    float zeroPoint = config->zeroPoint;
    int j = 0;
#ifdef __aarch64__
    float32x4_t scales = vdupq_n_f32(scale);
    float32x4_t zeros = vdupq_n_f32(zeroPoint + 0.5);
    int32x4_t maxds = vcombine_s32(vcreate_s32(0x000000ff000000ff), vcreate_s32(0x000000ff000000ff));
    int32x4_t minds = vcombine_s32(vcreate_s32(0x0000000000000000), vcreate_s32(0x0000000000000000));
    for (; j + 7 < len; j += 8) {
        float32x4_t fin1 = vld1q_f32(fValue + j);
        float32x4_t fin2 = vld1q_f32(fValue + j + 4);
        fin1 = vaddq_f32(vdivq_f32(fin1, scales), zeros);
        fin2 = vaddq_f32(vdivq_f32(fin2, scales), zeros);
        int32x4_t out1 = vcvtq_s32_f32(fin1);
        int32x4_t out2 = vcvtq_s32_f32(fin2);
        out1 = vmaxq_s32(out1, minds);
        out1 = vminq_s32(out1, maxds);
        out2 = vmaxq_s32(out2, minds);
        out2 = vminq_s32(out2, maxds);
        uint16x8_t out3 = vpaddq_u16(vreinterpretq_u16_s32(out1), vreinterpretq_u16_s32(out2));
        uint8x8_t out = vmovn_u16(out3);
        vst1_u8(uValue + j, out);
    }
#endif
#ifdef __AVX2__
    __m256 vScale = _mm256_set1_ps(scale);
    __m256 vZeroPoint = _mm256_set1_ps(zeroPoint);
    __m256 vZero = _mm256_setzero_ps();
    __m256 vHalf = _mm256_set1_ps(0.5f);
    __m256 vMax = _mm256_set1_ps(255.0f);
    for (; j + 7 < len; j += 8) {
        // Load 8 floats
        __m256 vValue = _mm256_loadu_ps(&fValue[j]);

        // fValue[j] / scale + zeroPoint + 0.5
        __m256 vScaled = _mm256_div_ps(vValue, vScale);
        __m256 vWithZP = _mm256_add_ps(vScaled, vZeroPoint);
        __m256 vWithHalf = _mm256_add_ps(vWithZP, vHalf);

        // max(..., 0.0)
        __m256 vClampedLow = _mm256_max_ps(vWithHalf, vZero);

        // min(..., 255.0)
        __m256 vClampedHigh = _mm256_min_ps(vClampedLow, vMax);

        // Convert to int32 (truncate)
        __m256i vInt32 = _mm256_cvtps_epi32(vClampedHigh);

        // Pack into 16-bit integers
        __m128i vInt16 = _mm_packus_epi32(_mm256_extractf128_si256(vInt32, 0), _mm256_extractf128_si256(vInt32, 1));

        // Pack into 8-bit integers
        __m128i vInt8 = _mm_packus_epi16(vInt16, vInt16);

        // Store the lower 64 bits (8 bytes)
        _mm_storel_epi64((__m128i *)&uValue[j], vInt8);
    }
#endif
    for (; j < len; j++) {
        uValue[j] = (uint8_t)(std::min(255., (double)std::max(fValue[j] / scale + zeroPoint + 0.5, 0.0)));
    }
}

MultiThreadOnlineQuantizationOp::MultiThreadOnlineQuantizationOp(float *input,
                                                                 uint8_t *output,
                                                                 LowBitConfig *configs,
                                                                 int n,
                                                                 int m,
                                                                 int group,
                                                                 int groupCnt,
                                                                 float *inputSums,
                                                                 float *iscales,
                                                                 float *izeros,
                                                                 int permuteType) {
    this->input = input;
    this->output = output;
    this->configs = configs;
    this->n = n;
    this->m = m;
    this->group = group;
    this->groupCnt = groupCnt;
    this->inputSums = inputSums;
    this->iscales = iscales;
    this->izeros = izeros;
    this->permuteType = permuteType;
}

void MultiThreadOnlineQuantizationOp::Run() {
    int realGroup = (this->m - 1) / this->groupCnt + 1;
    for (int i = 0; i < this->n; i++) {
        float *cur = this->input + i * this->m;
        uint8_t *u = this->output + i * this->m;
        for (int g = 0; g < realGroup; g++) {

            float minValue = 1e+9;
            float maxValue = 1e-9;

            int st = g * this->groupCnt;
            int end = std::min(this->m, (g + 1) * this->groupCnt);
            GetArrayMinMax(cur + st, end - st, minValue, maxValue);
            this->configs[i * realGroup + g] = LowBitConfig(maxValue, minValue, 0, 8);
            QuantizationAll(cur + st, u + st, end - st, &(this->configs[i * realGroup + g]));
        }
    }

    if (permuteType == 0) {
        // for INT8 * INT8
#ifdef __AVX2__
        for (int i = 0; i < n * m; i++) {
            output[i] = (output[i] + !output[i]);
        }
#endif
    }

    if (permuteType == 1) {
        // for INT8 * INT4
#ifdef __AVX2__
        Avx2InputPermute(output, n, m);
#endif
    }

    if (inputSums != nullptr) {
        for (int i = 0; i < n; i++) {
            for (int g = 0; g < realGroup; g++) {
                iscales[i * group + g] = configs[i * group + g].scale;
                izeros[i * group + g] = configs[i * group + g].zeroPoint;
                int sum = 0;
                int j = g * groupCnt;
#ifdef __AVX2__
                const __m256i ones8 = _mm256_set1_epi8(1);
                const __m256i ones16 = _mm256_set1_epi16(1);
                __m256i acc = _mm256_setzero_si256();
                for (; j + 31 < (g + 1) * groupCnt && j + 31 < m; j += 32) {
                    __m256i data = _mm256_loadu_si256((__m256i *)(output + i * m + j));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(data, ones8), ones16));
                }
                sum += I32sum(acc);
#endif
                for (; j < (g + 1) * groupCnt && j < m; j++) {
                    sum += output[i * m + j];
                }
                inputSums[i * group + g] = sum;
            }
        }
    }

    if (permuteType == 0) {
        // for INT8 * INT8
#ifdef __AVX2__
        for (int i = 0; i < n * m; i++) {
            output[i] ^= 128;
        }
#endif
    }
}

void OnlineQuantization(float *inputData,
                        std::vector<uint8_t> &uinput,
                        std::vector<LowBitConfig> &inputConfigs,
                        int n,
                        int m,
                        int group,
                        int groupCnt,
                        std::vector<float> &inputSums,
                        std::vector<float> &iscales,
                        std::vector<float> &izeros,
                        int permuteType) {
    inputConfigs.resize(n * group);
    uinput.resize(n * m);
    inputSums.resize(n * group);
    iscales.resize(n * group);
    izeros.resize(n * group);

    if (n > 1) {
        auto pool = GetAlivePool();
        int threadNum = pool->threads.size();
        int per = n / pool->threads.size();
        int cur = 0;
        std::vector<MultiThreadOnlineQuantizationOp *> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new MultiThreadOnlineQuantizationOp(inputData + cur * m,
                                                              uinput.data() + cur * m,
                                                              inputConfigs.data() + cur * group,
                                                              end - cur,
                                                              m,
                                                              group,
                                                              groupCnt,
                                                              inputSums.data() + cur * group,
                                                              iscales.data() + cur * group,
                                                              izeros.data() + cur * group,
                                                              permuteType));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    } else {
        MultiThreadOnlineQuantizationOp(
            inputData, uinput.data(), inputConfigs.data(), n, m, group, groupCnt, inputSums.data(), iscales.data(), izeros.data(), permuteType)
            .Run();
    }
}

MultiThreadLinearInt4NoZeroOp::MultiThreadLinearInt4NoZeroOp(uint8_t *a,
                                                             uint8_t *b,
                                                             int32_t *c,
                                                             int n,
                                                             int m,
                                                             int k,
                                                             int kstride,
                                                             int *weightSums,
                                                             float *weightMins,
                                                             float *scales,
                                                             float *bias,
                                                             LowBitConfig *config,
                                                             float *inputSums) {
    this->a = a;
    this->b = b;
    this->c = c;
    this->n = n;
    this->m = m;
    this->k = k;
    this->kstride = kstride;
    this->weightSums = weightSums;
    this->weightMins = weightMins;
    this->scales = scales;
    this->bias = bias;
    this->config = config;
    this->inputSums = inputSums;
}

void MultiThreadLinearInt4NoZeroOp::Run() {
#ifdef __ARM_FEATURE_DOTPROD
#define RUNBLOCK(x)                                                                                                                                  \
    for (; block + (x - 1) < n; block += (x))                                                                                                        \
        RunSomeBlock(b, a + block * m, c, (x), sum, vi, block, k, m, kstride);
    int block = 0;
    uint32x2_t sum[16];
    uint8x8x2_t vi[16];
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
#else
    int block = 0;

    for (; block < n; block++) {
        uint8_t *weightWalk = b;
        uint8_t *inputStart = a + block * m;

        for (int i = 0; i < k; i++) {
            int value = 0;
            uint8_t *inputWalk = inputStart;
            int j = 0;
#ifdef __ARM_FEATURE_DOTPROD
            uint8x8_t maskHigh = vdup_n_u8(0xF0);
            uint8x8_t maskLow = vdup_n_u8(0xF);
            uint32x2_t sum0 = {0, 0};

            for (; j + 15 < m; j += 16) {
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

            for (; j + 15 < m; j += 16) {
                uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                uint8x8x2_t in = vld2_u8(inputWalk + j);
                uint8x8_t va = vand_u8(ori, maskLow);
                uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
            }
            value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
            value += DotU4U8(weightWalk + i * m / 2, inputWalk, m);
            j += m;
#endif

            for (; j + 1 < m; j += 2) {
                int id = (i * m + j) / 2;
                value += (weightWalk[id] >> 4) * inputWalk[j];
                value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
            }

            c[block * kstride + i] = value;
        }
    }
#endif
    for (int block = 0; block < n; block++) {
        for (int i = 0; i < k; i++) {
            int value = c[block * kstride + i];
            value -= weightSums[i] * config[block].zeroPoint;
            ((float *)c)[block * kstride + i] = scales[i] * config[block].scale * value +
                                                weightMins[i] * ((float)inputSums[block] - (int)config[block].zeroPoint * m) * config[block].scale +
                                                (bias == nullptr ? 0.0 : bias[i]);
        }
    }
};

// a = [n, m], b = [k, m], c = aT(b') = [n, k]
void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a,
                                        uint8_t *b,
                                        float *c,
                                        int n,
                                        int m,
                                        int k,
                                        int *weightSums,
                                        float *weightMins,
                                        float *scales,
                                        float *bias,
                                        std::vector<float> &inputSums,
                                        std::vector<float> &iscales,
                                        std::vector<float> &izeros,
                                        std::vector<LowBitConfig> &configs,
                                        int startTid,
                                        int threadNum,
                                        int group,
                                        int groupCnt,
                                        std::vector<MultiThreadBaseOp *> &ops,
                                        AliveThreadPool *pool) {
    int per = k / threadNum;
    int cur = 0;

    for (int i = 0; i < threadNum; i++) {
        int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
        if (group > 1) {
            ops[startTid + i] = new MultiThreadLinearInt8Int4GroupOp(a,
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
                                                                     iscales.data(),
                                                                     izeros.data(),
                                                                     inputSums.data(),
                                                                     group,
                                                                     groupCnt);
        } else {
            ops[startTid + i] = new MultiThreadLinearInt4NoZeroOp(a,
                                                                  b + cur * m / 2,
                                                                  (int32_t *)c + cur,
                                                                  n,
                                                                  m,
                                                                  end - cur,
                                                                  k,
                                                                  weightSums + cur * group,
                                                                  weightMins + cur * group,
                                                                  scales + cur * group,
                                                                  (bias == nullptr ? (float *)nullptr : bias + cur),
                                                                  configs.data(),
                                                                  inputSums.data());
        }
        cur = end;
    }
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(startTid + i, ops[startTid + i]);
    }
}

void CpuSplitOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
    int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

    int dimslen = input.dims.size();
    axis = (axis % dimslen + dimslen) % dimslen;

    std::vector<int> dims = input.dims;
    start = std::max(0, std::min(dims[axis] - 1, start));
    end = std::max(0, std::min(dims[axis], end));

    dims[axis] = end - start;
    output.dataType = input.dataType;
    output.Resize(dims);
}

MultiThreadSliceOp::MultiThreadSliceOp(uint8_t *output, uint8_t *input, int outer, int outputStride, int inputStride, int copyLen) {
    this->output = output;
    this->input = input;
    this->outputStride = outputStride;
    this->inputStride = inputStride;
    this->copyLen = copyLen;
}

void MultiThreadSliceOp::Run() {
    for (int o = 0; o < outer; o++) {
        memcpy(output + o * outputStride, input + o * inputStride, copyLen);
    }
}

static void RunMultiThreadSlice(uint8_t *output, uint8_t *input, int outer, int inputStride, int outputStride, int copyLen, AliveThreadPool *pool) {
    if (outer == 1) {
        (MultiThreadSliceOp(output, input, outer, outputStride, inputStride, copyLen)).Run();
        return;
    }
    int threadNum = pool->threads.size();
    int per = outer / pool->threads.size();
    int cur = 0;
    std::vector<MultiThreadSliceOp *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
        ops.push_back(new MultiThreadSliceOp(output + cur * outputStride, input + cur * inputStride, end - cur, outputStride, inputStride, copyLen));
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

void CpuSplitOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
    int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

    output.Allocate();

    int dimslen = input.dims.size();
    axis = (axis % dimslen + dimslen) % dimslen;

    std::vector<int> dims = input.dims;
    start = std::max(0, std::min(dims[axis] - 1, start));
    end = std::max(0, std::min(dims[axis], end));

    int outer = input.Count(0) / input.Count(axis);
    int inputStride = input.Count(axis);
    int outputStride = output.Count(axis);
    int inner = input.strides[axis];
    int unitSize = input.unitSize;

    RunMultiThreadSlice(output.cpuData,
                        input.cpuData + start * inner * unitSize,
                        outer,
                        inputStride * unitSize,
                        outputStride * unitSize,
                        (end - start) * inner * unitSize,
                        GetAlivePool());
}

void CpuRepeatOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    std::vector<int> dims = input.dims;
    dims[axis] *= repeatTimes;

    output.dataType = input.dataType;
    output.Resize(dims);
}

void CpuRepeatOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;

    output.Allocate();

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.strides[axis];
    int inputStride = input.Count(axis);
    int outputStride = output.Count(axis);
    int unitSize = input.unitSize;

    for (int o = 0; o < outer; o++) {
        for (int t = 0; t < repeatTimes; t++) {
            std::memcpy(output.cpuData + o * outputStride * unitSize + t * channels * inner * unitSize,
                        input.cpuData + o * inputStride * unitSize,
                        channels * inner * unitSize);
        }
    }
}

void CpuCatOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    Data &output = *(datas.find("output")->second);

    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

    if (input0.dims.size() == 0 && input1.dims.size() > 0) {
        output.Resize(input1.dims);
        return;
    }
    if (input1.dims.size() == 0 && input0.dims.size() > 0) {
        output.Resize(input0.dims);
        return;
    }

    AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                    "Cat's input's type should be float32 or float16.\n");
    AssertInFastLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.");

    int dimsLen = input0.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    for (int i = 0; i < dimsLen; i++) {
        if (i != axis) {
            AssertInFastLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
        }
    }

    std::vector<int> dims = input0.dims;
    dims[axis] += input1.dims[axis];

    output.dataType = input0.dataType;
    output.Resize(dims);
}

void CpuCatOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    Data &output = *(datas.find("output")->second);

    output.Allocate();

    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

    if (input0.dims.size() == 0 && input1.dims.size() > 0) {
        output.CopyFrom(input1);
        return;
    }

    if (input1.dims.size() == 0 && input0.dims.size() > 0) {
        output.CopyFrom(input0);
        return;
    }

    int dimsLen = input0.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    int outer = output.Count(0) / output.Count(axis);
    int input0Stride = input0.Count(axis);
    int input1Stride = input1.Count(axis);
    int outputStride = output.Count(axis);
    int unitSize = input0.unitSize;

    for (int o = 0; o < outer; o++) {
        std::memcpy(output.cpuData + o * outputStride * unitSize, input0.cpuData + o * input0Stride * unitSize, input0Stride * unitSize);
        std::memcpy(output.cpuData + o * outputStride * unitSize + input0Stride * unitSize,
                    input1.cpuData + o * input1Stride * unitSize,
                    input1Stride * unitSize);
    }
}

void DoCpuCatDirect(Data &input0, Data &input1, int axis) {
    AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                    "CatDirect's input's type should be float32 or float16.\n");
    AssertInFastLLM(input0.dataDevice == input1.dataDevice, "CatDirect error: inputs should use same device.\n");

    if (input0.dims.size() == 0) {
        input0.Resize(input1.dims);
        AssertInFastLLM(input0.expansionDims.size() == input1.dims.size() && input1.dims[axis] <= input0.expansionDims[axis],
                        "CatDirect Error: input0's expansion size is not enough.\n");
        int outer = input1.Count(0) / input1.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;
        for (int o = 0; o < outer; o++) {
            memcpy(input0.cpuData + o * input0Stride * unitSize, input1.cpuData + o * input1Stride * unitSize, input1.dims[axis] * inner * unitSize);
        }

        return;
    }

    std::vector<int> dims = input0.dims;
    std::vector<int> oldDims = dims;
    dims[axis] += input1.dims[axis];
    input0.Resize(dims);
    int outer = input0.Count(0) / input0.Count(axis);
    int input0Stride = input0.Count(axis);
    int input1Stride = input1.Count(axis);

    int inner = input0.strides[axis];
    int unitSize = input0.unitSize;

    for (int o = 0; o < outer; o++) {
        memcpy(input0.cpuData + o * input0Stride * unitSize + oldDims[axis] * inner * unitSize,
               input1.cpuData + (o * input1Stride) * unitSize,
               input1.dims[axis] * inner * unitSize);
    }
}

void CpuCatDirectOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);

    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    DoCpuCatDirect(input0, input1, axis);
}

MultiThreadMatMulSingleOp::MultiThreadMatMulSingleOp(float *input0Base,
                                                     float *input1Base,
                                                     float *outputBase,
                                                     int input0Spatial,
                                                     int input1Spatial,
                                                     int outputSpatial,
                                                     int input0Stride,
                                                     int input1Stride,
                                                     int n,
                                                     int m,
                                                     int k,
                                                     float alpha,
                                                     int st,
                                                     int end) {
    this->input0Base = input0Base;
    this->input1Base = input1Base;
    this->outputBase = outputBase;
    this->input0Spatial = input0Spatial;
    this->input1Spatial = input1Spatial;
    this->outputSpatial = outputSpatial;
    this->input0Stride = input0Stride;
    this->input1Stride = input1Stride;
    this->n = n;
    this->m = m;
    this->k = k;
    this->alpha = alpha;
    this->st = st;
    this->end = end;
}

void MultiThreadMatMulSingleOp::Run() {
    for (int b = this->st; b < this->end; b++) {
        float *inputData0 = this->input0Base + b * this->input0Spatial;
        float *inputData1 = this->input1Base + b * this->input1Spatial;
        float *outputData = this->outputBase + b * this->outputSpatial;

        std::fill(outputData, outputData + n * k, 0.0f);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                float now = inputData0[i * this->input0Stride + j] * alpha;
                for (int l = 0; l < k; l++) {
                    outputData[i * k + l] += now * inputData1[j * k + l];
                }
            }
        }
    }
}

MultiThreadMatMulFloat16SingleOp::MultiThreadMatMulFloat16SingleOp(uint16_t *input0Base,
                                                                   uint16_t *input1Base,
                                                                   uint16_t *outputBase,
                                                                   int input0Spatial,
                                                                   int input1Spatial,
                                                                   int outputSpatial,
                                                                   int input0Stride,
                                                                   int input1Stride,
                                                                   int n,
                                                                   int m,
                                                                   int k,
                                                                   float alpha,
                                                                   int st,
                                                                   int end) {
    this->input0Base = input0Base;
    this->input1Base = input1Base;
    this->outputBase = outputBase;
    this->input0Spatial = input0Spatial;
    this->input1Spatial = input1Spatial;
    this->outputSpatial = outputSpatial;
    this->input0Stride = input0Stride;
    this->input1Stride = input1Stride;
    this->n = n;
    this->m = m;
    this->k = k;
    this->alpha = alpha;
    this->st = st;
    this->end = end;
}

void MultiThreadMatMulFloat16SingleOp::Run() {
    float *input0 = new float[n * m];
    float *input1 = new float[m * k];
    float *output = new float[n * k];

    for (int b = st; b < end; b++) {
        uint16_t *input0Data = input0Base + b * input0Spatial;
        uint16_t *input1Data = input1Base + b * input1Spatial;
        uint16_t *outputData = outputBase + b * outputSpatial;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                input0[i * m + j] = g_fp16ToFp32Manager.dict[input0Data[i * input0Stride + j]];
            }
        }
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < k; l++) {
                input1[j * k + l] = g_fp16ToFp32Manager.dict[input1Data[j * k + l]];
            }
        }
        std::fill(output, output + n * k, 0.0f);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                float now = input0[i * m + j] * alpha;
                for (int l = 0; l < k; l++) {
                    output[i * k + l] += (now * input1[j * k + l]);
                }
            }
        }
        for (int i = 0; i < n * k; i++) {
            outputData[i] = float_to_half(output[i]);
        }
    }

    delete[] input0;
    delete[] input1;
    delete[] output;
}

MultiThreadMatMulTransBSingleOp::MultiThreadMatMulTransBSingleOp(float *input0Base,
                                                                 float *input1Base,
                                                                 float *outputBase,
                                                                 int input0Spatial,
                                                                 int input1Spatial,
                                                                 int outputSpatial,
                                                                 int input0Stride,
                                                                 int input1Stride,
                                                                 int n,
                                                                 int m,
                                                                 int k,
                                                                 float alpha,
                                                                 int st,
                                                                 int end) {
    this->input0Base = input0Base;
    this->input1Base = input1Base;
    this->outputBase = outputBase;
    this->input0Spatial = input0Spatial;
    this->input1Spatial = input1Spatial;
    this->outputSpatial = outputSpatial;
    this->input0Stride = input0Stride;
    this->input1Stride = input1Stride;
    this->n = n;
    this->m = m;
    this->k = k;
    this->alpha = alpha;
    this->st = st;
    this->end = end;
}

void MultiThreadMatMulTransBSingleOp::Run() {
    for (int b = st; b < end; b++) {
        float *input0Data = input0Base + b * input0Spatial;
        float *input1Data = input1Base + b * input1Spatial;
        float *outputData = outputBase + b * outputSpatial;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                float now = 0.0f;
                int l = 0;
#ifdef __aarch64__
                float32x4_t sum = {0, 0, 0, 0};
                for (; l + 3 < m; l += 4) {
                    sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(input0Data + i * input0Stride + l), vld1q_f32(input1Data + j * input1Stride + l)));
                }
                now += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                __m256 vsum = _mm256_set1_ps(0.0f);
                for (; l + 7 < m; l += 8) {
                    __m256 vx = _mm256_loadu_ps((const float *)(input0Data + i * input0Stride + l));
                    __m256 vy = _mm256_loadu_ps((const float *)(input1Data + j * input1Stride + l));
                    vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                }
                now += Floatsum(vsum);
#endif
                for (; l < m; l++) {
                    now += input0Data[i * input0Stride + l] * input1Data[j * input1Stride + l];
                }
                outputData[i * k + j] = now * alpha;
            }
        }
    }
}

MultiThreadMatMulTransBFloat16SingleOp::MultiThreadMatMulTransBFloat16SingleOp(uint16_t *input0Base,
                                                                               uint16_t *input1Base,
                                                                               uint16_t *outputBase,
                                                                               int input0Spatial,
                                                                               int input1Spatial,
                                                                               int outputSpatial,
                                                                               int input0Stride,
                                                                               int input1Stride,
                                                                               int n,
                                                                               int m,
                                                                               int k,
                                                                               float alpha,
                                                                               int st,
                                                                               int end) {
    this->input0Base = input0Base;
    this->input1Base = input1Base;
    this->outputBase = outputBase;
    this->input0Spatial = input0Spatial;
    this->input1Spatial = input1Spatial;
    this->outputSpatial = outputSpatial;
    this->input0Stride = input0Stride;
    this->input1Stride = input1Stride;
    this->n = n;
    this->m = m;
    this->k = k;
    this->alpha = alpha;
    this->st = st;
    this->end = end;
}

void MultiThreadMatMulTransBFloat16SingleOp::Run() {
    for (int b = st; b < end; b++) {
        uint16_t *input0Data = input0Base + b * input0Spatial;
        uint16_t *input1Data = input1Base + b * input1Spatial;
        uint16_t *outputData = outputBase + b * outputSpatial;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                float now = 0.0f;
                int l = 0;
#if defined(__AVX__)
                __m256 vsum = _mm256_set1_ps(0.0f);
                for (; l + 7 < m; l += 8) {
                    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(input0Data + i * input0Stride + l)));
                    __m256 vy = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(input1Data + j * input1Stride + l)));
                    vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                }
                now += Floatsum(vsum);
#endif
                for (; l < m; l++) {
                    now += g_fp16ToFp32Manager.dict[input0Data[i * input0Stride + l]] * g_fp16ToFp32Manager.dict[input1Data[j * input1Stride + l]];
                }
                outputData[i * k + j] = float_to_half(now * alpha);
            }
        }
    }
}

void CpuMatMulOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    Data &output = *(datas.find("output")->second);

    AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMul error: inputs should use same device.\n");
    AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
                    "MatMul's input's type should be float32 or float16.\n");
    AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2, "MatMul's input's shape's size should be >= 2.\n");
    AssertInFastLLM(input0.dims.back() == input1.dims[input1.dims.size() - 2], "MatMul's shape error.\n");
    int input0Spatial = input0.Count(input0.dims.size() - 2);
    int input1Spatial = input1.Count(input1.dims.size() - 2);
    int batch0 = input0.Count(0) / input0Spatial;
    int batch1 = input1.Count(0) / input1Spatial;
    int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
    AssertInFastLLM(batch0 == batch1 * group, "MatMul: input0.dims[1] should be equal to input1.dims[0] * group.\n");
    // AssertInFastLLM(batch0 == batch1, "MatMul's shape error.\n");

    std::vector<int> dims = input0.dims;
    dims.back() = input1.dims[input1.dims.size() - 1];

    output.dataType = input0.dataType;
    output.Resize(dims);
}

void CpuMatMulOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    Data &output = *(datas.find("output")->second);

    output.Allocate();

    float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
    int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
    int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
    int input1Spatial = input1.Count(input1.dims.size() - 2);
    int input0Stride = input0.strides[input0.dims.size() - 2];
    int input1Stride = input1.strides[input1.dims.size() - 2];
    int n = input0.dims[input0.dims.size() - 2] * group;
    int m = input0.dims.back();
    int k = input1.dims[input1.dims.size() - 1];
    int batch0 = input0.Count(0) / input0Spatial;
    int batch1 = input1.Count(0) / input1Spatial;

    int outputSpatial = output.Count(output.dims.size() - 2) * group;
    int threadNum = GetThreads();
#ifdef _WIN64
    threadNum = 1;
#endif
    if (batch0 * n * m * k < 64 * 4096) {
        threadNum = 1;
    }
    threadNum = std::min(threadNum, 4);
    // TODO: 汇编优化
    int per = batch0 / threadNum;
    int cur = 0;
    if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
        auto *pool = GetAlivePool();
        int threads = pool->threads.size();
        std::vector<MultiThreadMatMulSingleOp *> ops;
        for (int o = 0; o < batch0; o++) {
            ops.push_back(new MultiThreadMatMulSingleOp((float *)input0.cpuData,
                                                        (float *)input1.cpuData,
                                                        (float *)output.cpuData,
                                                        input0Spatial,
                                                        input1Spatial,
                                                        outputSpatial,
                                                        input0Stride,
                                                        input1Stride,
                                                        n,
                                                        m,
                                                        k,
                                                        alpha,
                                                        o,
                                                        o + 1));
        }
        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
    } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
        std::vector<uint16_t> fp16InputData;
        fp16InputData.resize(input0.Count(0));
        Float32ToFloat16((float *)input0.cpuData, fp16InputData.data(), input0.Count(0));

        auto *pool = GetAlivePool();
        int threads = pool->threads.size();
        std::vector<MultiThreadMatMulFloat16SingleOp *> ops;
        for (int o = 0; o < batch0; o++) {
            ops.push_back(new MultiThreadMatMulFloat16SingleOp((uint16_t *)fp16InputData.data(),
                                                               (uint16_t *)input1.cpuData,
                                                               (uint16_t *)output.cpuData,
                                                               input0Spatial,
                                                               input1Spatial,
                                                               outputSpatial,
                                                               input0Stride,
                                                               input1Stride,
                                                               n,
                                                               m,
                                                               k,
                                                               alpha,
                                                               o,
                                                               o + 1));
        }
        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
    } else if (input0.dataType == DataType::FLOAT16) {
        auto *pool = GetAlivePool();
        int threads = pool->threads.size();
        std::vector<MultiThreadMatMulFloat16SingleOp *> ops;
        if (batch0 == 1) {
            int partn = std::max(1, n / threads);
            for (int o = 0; o < n; o += partn) {
                int len = std::min(partn, n - o);
                ops.push_back(new MultiThreadMatMulFloat16SingleOp(((uint16_t *)input0.cpuData) + o * m,
                                                                   (uint16_t *)input1.cpuData,
                                                                   ((uint16_t *)output.cpuData) + o * k,
                                                                   input0Spatial,
                                                                   input1Spatial,
                                                                   outputSpatial,
                                                                   input0Stride,
                                                                   input1Stride,
                                                                   len,
                                                                   m,
                                                                   k,
                                                                   alpha,
                                                                   0,
                                                                   1));
            }
        } else {
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulFloat16SingleOp((uint16_t *)input0.cpuData,
                                                                   (uint16_t *)input1.cpuData,
                                                                   (uint16_t *)output.cpuData,
                                                                   input0Spatial,
                                                                   input1Spatial,
                                                                   outputSpatial,
                                                                   input0Stride,
                                                                   input1Stride,
                                                                   n,
                                                                   m,
                                                                   k,
                                                                   alpha,
                                                                   o,
                                                                   o + 1));
            }
        }
        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
    }
}

void CpuMatMulTransBOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    Data &output = *(datas.find("output")->second);

    AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMulTransB error: inputs should use same device.\n");
    AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
                    "MatMulTransB's input's type should be float32 or float16.\n");
    AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2, "MatMulTransB's input's shape's size should be >= 2.\n");
    AssertInFastLLM(input0.dims.back() == input1.dims.back(), "MatMulTransB's shape error.\n");
    int input0Spatial = input0.Count(input0.dims.size() - 2);
    int input1Spatial = input1.Count(input1.dims.size() - 2);
    int batch0 = input0.Count(0) / input0Spatial;
    int batch1 = input1.Count(0) / input1Spatial;
    int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
    AssertInFastLLM(batch0 == batch1 * group, "MatMulTransB: input0.dims[0] should be equal to input1.dims[0] * group.\n");
    // AssertInFastLLM(batch0 == batch1, "MatMulTransB's shape error.\n");

    std::vector<int> dims = input0.dims;
    dims.back() = input1.dims[input1.dims.size() - 2];
    output.dataType = input0.dataType;
    output.Resize(dims);
}

void CpuMatMulTransBOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    Data &output = *(datas.find("output")->second);

    output.Allocate();

    float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;
    int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
    int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
    int input1Spatial = input1.Count(input1.dims.size() - 2);
    int input0Stride = input0.strides[input0.dims.size() - 2];
    int input1Stride = input1.strides[input1.dims.size() - 2];
    int n = input0.dims[input0.dims.size() - 2] * group;
    int m = input0.dims.back();
    int k = input1.dims[input1.dims.size() - 2];
    int batch0 = input0.Count(0) / input0Spatial;
    int batch1 = input1.Count(0) / input1Spatial;

    int outputSpatial = output.Count(output.dims.size() - 2) * group;
    int threadNum = GetThreads();
#ifdef _WIN64
    threadNum = 1;
#endif
    if (batch0 * n * m * k < 64 * 4096) {
        threadNum = 1;
    }
    threadNum = std::min(threadNum, 4);
    int per = batch0 / threadNum;
    int cur = 0;
    if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
        auto *pool = GetAlivePool();
        int threads = pool->threads.size();
        std::vector<MultiThreadMatMulTransBSingleOp *> ops;
        for (int o = 0; o < batch0; o++) {
            ops.push_back(new MultiThreadMatMulTransBSingleOp((float *)input0.cpuData,
                                                              (float *)input1.cpuData,
                                                              (float *)output.cpuData,
                                                              input0Spatial,
                                                              input1Spatial,
                                                              outputSpatial,
                                                              input0Stride,
                                                              input1Stride,
                                                              n,
                                                              m,
                                                              k,
                                                              alpha,
                                                              o,
                                                              o + 1));
        }
        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
    } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
        std::vector<uint16_t> fp16InputData;
        fp16InputData.resize(input0.Count(0));
        Float32ToFloat16((float *)input0.cpuData, fp16InputData.data(), input0.Count(0));

        auto *pool = GetAlivePool();
        int threads = pool->threads.size();
        std::vector<MultiThreadMatMulTransBFloat16SingleOp *> ops;
        for (int o = 0; o < batch0; o++) {
            ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp((uint16_t *)fp16InputData.data(),
                                                                     (uint16_t *)input1.cpuData,
                                                                     (uint16_t *)output.cpuData,
                                                                     input0Spatial,
                                                                     input1Spatial,
                                                                     outputSpatial,
                                                                     input0Stride,
                                                                     input1Stride,
                                                                     n,
                                                                     m,
                                                                     k,
                                                                     alpha,
                                                                     o,
                                                                     o + 1));
        }
        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
    } else {
        auto *pool = GetAlivePool();
        int threads = pool->threads.size();
        std::vector<MultiThreadMatMulTransBFloat16SingleOp *> ops;
        if (batch0 == 1) {
            int partn = std::max(1, n / threads);
            for (int o = 0; o < n; o += partn) {
                int len = std::min(partn, n - o);
                ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(((uint16_t *)input0.cpuData) + o * m,
                                                                         (uint16_t *)input1.cpuData,
                                                                         ((uint16_t *)output.cpuData) + o * k,
                                                                         input0Spatial,
                                                                         input1Spatial,
                                                                         outputSpatial,
                                                                         input0Stride,
                                                                         input1Stride,
                                                                         len,
                                                                         m,
                                                                         k,
                                                                         alpha,
                                                                         0,
                                                                         1));
            }
        } else {
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp((uint16_t *)input0.cpuData,
                                                                         (uint16_t *)input1.cpuData,
                                                                         (uint16_t *)output.cpuData,
                                                                         input0Spatial,
                                                                         input1Spatial,
                                                                         outputSpatial,
                                                                         input0Stride,
                                                                         input1Stride,
                                                                         n,
                                                                         m,
                                                                         k,
                                                                         alpha,
                                                                         o,
                                                                         o + 1));
            }
        }
        for (int st = 0; st < ops.size(); st += threads) {
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < ops.size() && i < st + threads; i++) {
                pool->Wait(i - st);
            }
        }
    }
}

void CpuSiluOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Silu error: Data's type should be float32 or float16.\n");
    int len = input.Count(0);

    if (input.dataType == DataType::FLOAT16) {
        uint16_t *inputData = (uint16_t *)input.cpuData;
        uint16_t *outputData = (uint16_t *)output.cpuData;
        for (int i = 0; i < len; i++) {
            outputData[i] = g_fp16SiluManager.dict[inputData[i]];
        }
    } else {
        float *inputData = (float *)input.cpuData;
        float *outputData = (float *)output.cpuData;
        int i = 0;
#ifdef __aarch64__
        float32x4_t c1 = vdupq_n_f32(1.0f);
        for (; i + 3 < len; i += 4) {
            float32x4_t vx = vld1q_f32(inputData + i);
            float32x4_t vdiv = vaddq_f32(c1, exp_ps(vnegq_f32(vx)));
            vx = vdivq_f32(vx, vdiv);
            vst1q_f32(outputData + i, vx);
        }
#endif
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x / (1.0 + expf(-x));
        }
    }
}

void CpuTanHOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

    float temp = sqrt(2.0f / M_PI), factor = 0.044715;
    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;
    int len = input.Count(0);
    int i = 0;
    for (; i < len; i++) {
        outputData[i] = tanhf(inputData[i]);
    }
}

void CpuReluOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32, "Relu error: Data's type should be float32.\n");

    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;
    int len = input.Count(0);
    int i = 0;
    for (; i < len; i++) {
        float x = inputData[i];
        outputData[i] = x > 0 ? x : 0;
    }
}

void CpuSigmoidOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Sigmoid error: Data's type should be float32 or float16.\n");

    int len = input.Count(0);
    if (input.dataType == DataType::FLOAT16) {
        uint16_t *inputData = (uint16_t *)input.cpuData;
        uint16_t *outputData = (uint16_t *)output.cpuData;
        for (int i = 0; i < len; i++) {
            outputData[i] = g_fp16SigmoidManager.dict[inputData[i]];
        }
    } else {
        float *inputData = (float *)input.cpuData;
        float *outputData = (float *)output.cpuData;
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = 1.0 / (1.0 + exp(-x));
        }
    }
}

float erf(float a) {
    float r, s, t, u;

    t = fabsf(a);
    s = a * a;
    if (t > 0.927734375f) { // 475/512
        // maximum error 0.99527 ulp
        r = fmaf(-1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
        u = fmaf(-3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
        r = fmaf(r, s, u);
        r = fmaf(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
        r = fmaf(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
        r = fmaf(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
        r = fmaf(r, t, -t);
        r = 1.0f - expf(r);
        r = copysignf(r, a);
    } else {
        // maximum error 0.98929 ulp
        r = -5.96761703e-4f;             // -0x1.38e000p-11
        r = fmaf(r, s, 4.99119423e-3f);  //  0x1.471a58p-8
        r = fmaf(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
        r = fmaf(r, s, 1.12819925e-1f);  //  0x1.ce1c44p-4
        r = fmaf(r, s, -3.76125336e-1f); // -0x1.812700p-2
        r = fmaf(r, s, 1.28379166e-1f);  //  0x1.06eba8p-3
        r = fmaf(r, a, a);
    }
    return r;
}

void CpuGeluOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

    float temp = sqrt(2.0f / M_PI), factor = 0.044715;
    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;
    int len = input.Count(0);
    int i = 0;
    for (; i < len; i++) {
        float x = inputData[i];
        outputData[i] = x * 0.5f * (1.0f + erf(x / sqrt(2.0)));
    }
}

void CpuGeluNewOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;
    int len = input.Count(0);
    int i = 0;
#ifdef __aarch64__
    float32x4_t c0 = vdupq_n_f32(0.044715f);
    float32x4_t c1 = vdupq_n_f32(1.0f);
    float32x4_t c2 = vdupq_n_f32(0.7978845608028654f);
    float32x4_t c3 = vdupq_n_f32(0.5f);

    for (; i + 3 < len; i += 4) {
        float32x4_t vx = vld1q_f32(inputData + i);
        float32x4_t v1 = vaddq_f32(c1, vmulq_f32(vmulq_f32(c0, vx), vx));
        float32x4_t v2 = vmulq_f32(vmulq_f32(c2, vx), v1);
        float32x4_t vex = exp_ps(v2);
        float32x4_t venegx = exp_ps(vnegq_f32(v2));
        float32x4_t vtan = vdivq_f32(vsubq_f32(vex, venegx), vaddq_f32(vex, venegx));
        float32x4_t vout = vmulq_f32(vmulq_f32(c3, vx), vaddq_f32(c1, vtan));
        vst1q_f32(outputData + i, vout);
    }
#endif
#ifdef __AVX2__
    auto var1 = _mm256_set1_ps(0.044715f);
    auto var2 = _mm256_set1_ps(0.7978845608028654f);
    auto var3 = _mm256_set1_ps(378.f);
    auto var4 = _mm256_set1_ps(17325.f);
    auto var5 = _mm256_set1_ps(135135.f);
    auto var6 = _mm256_set1_ps(28.f);
    auto var7 = _mm256_set1_ps(3150.f);
    auto var8 = _mm256_set1_ps(62370.f);
    auto var9 = _mm256_set1_ps(135135.f);
    auto var10 = _mm256_set1_ps(0.5);
    auto varOne = _mm256_set1_ps(1.f);
    auto varNegOne = _mm256_set1_ps(-1.f);

    for (; i < len - 7; i += 8) {
        auto x = _mm256_loadu_ps(inputData + i);
        // sqrt(2 / PI) * (0.044715 * x^3 + x)
        auto y = _mm256_mul_ps(x, x);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var1);
        y = _mm256_add_ps(y, x);
        y = _mm256_mul_ps(y, var2);

        // y = tanh(y)
        {
            auto y2 = _mm256_mul_ps(y, y);
            auto w = _mm256_add_ps(y2, var3);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var4);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var5);
            w = _mm256_mul_ps(w, y);
            auto z = _mm256_mul_ps(y2, var6);
            z = _mm256_add_ps(z, var7);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var8);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var9);
            z = _mm256_div_ps(w, z);
            z = _mm256_max_ps(z, varNegOne);
            y = _mm256_min_ps(z, varOne);
        }

        y = _mm256_add_ps(y, varOne);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var10);
        _mm256_storeu_ps(outputData + i, y);
    }
#endif
    for (; i < len; i++) {
        float x = inputData[i];
        outputData[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }
}

void CpuSwigluOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);

    std::vector<int> dims = input.dims;
    dims[dims.size() - 1] /= 2;
    output.dataType = input.dataType;
    output.Resize(dims);
}

void CpuSwigluOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Swiglu error: Data's type should be float32 or float16.\n");
    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;

    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int outer = input.Count(0) / spatial;

    if (input.dataType == DataType::FLOAT16) {
        int len = input.Count(0);
        inputData = new float[len];
        outputData = new float[output.Count(0)];
        for (int i = 0; i < len; i++) {
            inputData[i] = g_fp16ToFp32Manager.dict[((uint16_t *)input.cpuData)[i]];
        }
    }

    for (int o = 0; o < outer; o++) {
        int i = 0;
#ifdef __aarch64__
        float32x4_t c1 = vdupq_n_f32(1.0f);
        for (; i + 3 < mid; i += 4) {
            float32x4_t vx = vld1q_f32(inputData + i);
            float32x4_t vy = vld1q_f32(inputData + i + mid);
            vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
            vy = vmulq_f32(vx, vy);
            vst1q_f32(outputData + i, vy);
        }
#endif

#ifdef __AVX2__X
        for (; i + 7 < mid; i += 8) { // Process 8 elements at a time
            // Load x values (inputData[i..i+7]) and y values (inputData[i+mid..i+mid+7])
            __m256 x = _mm256_loadu_ps(&inputData[i]);
            __m256 y = _mm256_loadu_ps(&inputData[i + mid]);

            // Compute sigmoid: 1.0 / (1.0 + expf(-x))
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            __m256 exp_neg_x = exp256_ps(neg_x); // See note below about exp_ps
            __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
            __m256 sigmoid = _mm256_div_ps(x, denom);

            // Multiply by y and store result
            __m256 result = _mm256_mul_ps(sigmoid, y);
            _mm256_storeu_ps(&outputData[i], result);
        }
#endif
        for (; i < mid; i++) {
            float x = inputData[i], y = inputData[i + mid];
            outputData[i] = (x / (1.0 + expf(-x))) * y;
        }
        inputData += spatial;
        outputData += spatial / 2;
    }

    if (input.dataType == DataType::FLOAT16) {
        inputData -= input.Count(0);
        outputData -= output.Count(0);
        int len = output.Count(0);
        for (int i = 0; i < len; i++) {
            ((uint16_t *)output.cpuData)[i] = float_to_half(outputData[i]);
        }

        delete[] inputData;
        delete[] outputData;
    }
}

MultiThreadSwigluOp::MultiThreadSwigluOp(float *input, int mid, int len, float *output, int n, int inputStride, int outputStride) {
    this->input = input;
    this->output = output;
    this->n = n;
    this->len = len;
    this->inputStride = inputStride;
    this->outputStride = outputStride;
    this->mid = mid;
}

void MultiThreadSwigluOp::Run() {
    for (int o = 0; o < n; o++) {
        float *cur = (float *)input + o * inputStride;
        float *out = (float *)output + o * outputStride;
        int i = 0;
#ifdef __aarch64__
        float32x4_t c1 = vdupq_n_f32(1.0f);
        for (; i + 3 < len; i += 4) {
            float32x4_t vx = vld1q_f32(cur + i);
            float32x4_t vy = vld1q_f32(cur + i + mid);
            vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
            vy = vmulq_f32(vx, vy);
            vst1q_f32(out + i, vy);
        }
#endif
#ifdef __AVX2__
        for (; i + 7 < len; i += 8) { // Process 8 elements at a time
            // Load x values (inputData[i..i+7]) and y values (inputData[i+mid..i+mid+7])
            __m256 x = _mm256_loadu_ps(&cur[i]);
            __m256 y = _mm256_loadu_ps(&cur[i + mid]);

            // Compute sigmoid: 1.0 / (1.0 + expf(-x))
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            __m256 exp_neg_x = exp256_ps(neg_x); // See note below about exp_ps
            __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
            __m256 sigmoid = _mm256_div_ps(x, denom);

            // Multiply by y and store result
            __m256 result = _mm256_mul_ps(sigmoid, y);
            _mm256_storeu_ps(&out[i], result);
        }
#endif
        for (; i < len; i++) {
            float x = cur[i], y = cur[i + mid];
            out[i] = (x / (1.0 + expf(-x))) * y;
        }
    }
}

MultiThreadSwigluFloat16Op::MultiThreadSwigluFloat16Op(
    uint16_t *input, int mid, int len, uint16_t *output, int n, int inputStride, int outputStride) {
    this->input = input;
    this->output = output;
    this->n = n;
    this->len = len;
    this->inputStride = inputStride;
    this->outputStride = outputStride;
    this->mid = mid;
}

void MultiThreadSwigluFloat16Op::Run() {
    for (int o = 0; o < n; o++) {
        uint16_t *cur = (uint16_t *)input + o * inputStride;
        uint16_t *out = (uint16_t *)output + o * outputStride;
        int i = 0;
#ifdef __AVX2__
        for (; i + 7 < len; i += 8) { // Process 8 elements at a time
            __m128i x_half = _mm_loadu_si128((const __m128i *)&cur[i]);
            __m256 x = _mm256_cvtph_ps(x_half); // Convert float16 to float32

            // Load 8 float16 values from cur[i+mid..i+mid+7] and convert to float32
            __m128i y_half = _mm_loadu_si128((const __m128i *)&cur[i + mid]);
            __m256 y = _mm256_cvtph_ps(y_half); // Convert float16 to float32

            // Compute sigmoid: 1.0 / (1.0 + expf(-x))
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            __m256 exp_neg_x = exp256_ps(neg_x); // See note below about exp_ps
            __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
            __m256 sigmoid = _mm256_div_ps(x, denom);

            // Multiply by y and store result
            __m256 result = _mm256_mul_ps(sigmoid, y);

            // Convert result back to float16 and store
            __m128i result_half = _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)&out[i], result_half);
        }
#endif
        for (; i < len; i++) {
            float x = g_fp16ToFp32Manager.dict[cur[i]], y = g_fp16ToFp32Manager.dict[cur[i + mid]];
            out[i] = float_to_half((x / (1.0 + expf(-x))) * y);
        }
    }
}

void DoCpuSwigluReshape(Data &input, Data &output) {
    std::vector<int> dims = input.dims;
    dims[dims.size() - 1] /= 2;
    output.dataType = input.dataType;
    output.Resize(dims);
}

void DoCpuSwiglu(Data &input, Data &output) {
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Swiglu error: Data's type should be float32 or float16.\n");

    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;

    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int outer = input.Count(0) / spatial;

    if (input.dataType == DataType::FLOAT32) {
        (MultiThreadSwigluOp((float *)inputData, spatial / 2, spatial / 2, (float *)outputData, outer, spatial, spatial / 2)).Run();
    } else if (input.dataType == DataType::FLOAT16) {
        (MultiThreadSwigluFloat16Op((uint16_t *)inputData, spatial / 2, spatial / 2, (uint16_t *)outputData, outer, spatial, spatial / 2)).Run();
    } else {
        printf("Unsupport swiglu type.");
    }
}

void CpuMulOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();

    float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Mul error: Data's type should be float32 or float16.\n");

    int len = input.Count(0);

    if (input.dataType == DataType::FLOAT32) {
        float *inputData = (float *)input.cpuData;
        float *outputData = (float *)output.cpuData;
        for (int i = 0; i < len; i++) {
            outputData[i] = inputData[i] * v;
        }
    } else if (input.dataType == DataType::FLOAT16) {
        uint16_t *inputData = (uint16_t *)input.cpuData;
        uint16_t *outputData = (uint16_t *)output.cpuData;
        for (int i = 0; i < len; i++) {
            outputData[i] = float_to_half(g_fp16ToFp32Manager.dict[inputData[i]] * v);
        }
    }
}

void CpuAddOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();

    float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Add error: Data's type should be float32 or float16.\n");

    int len = input.Count(0);

    if (input.dataType == DataType::FLOAT32) {
        float *inputData = (float *)input.cpuData;
        float *outputData = (float *)output.cpuData;
        for (int i = 0; i < len; i++) {
            outputData[i] = inputData[i] + v;
        }
    } else if (input.dataType == DataType::FLOAT16) {
        uint16_t *inputData = (uint16_t *)input.cpuData;
        uint16_t *outputData = (uint16_t *)output.cpuData;
        for (int i = 0; i < len; i++) {
            outputData[i] = float_to_half(g_fp16ToFp32Manager.dict[inputData[i]] + v);
        }
    }
}

void CpuMulToOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    AssertInFastLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");

    int len = input0.Count(0);
    int inner = input1.Count(0);
    AssertInFastLLM(len % inner == 0, "MulTo error: Data`s shape can`t perform MulTo operation.\n");
    int round = (len / inner);

    if (input0.dataType == DataType::FLOAT16) {
        uint16_t *input0Data = (uint16_t *)input0.cpuData;
        uint16_t *input1Data = (uint16_t *)input1.cpuData;
        for (int j = 0; j < round; j++) {
            for (int i = 0; i < len; i++) {
                input0Data[i] = float_to_half(g_fp16ToFp32Manager.dict[input0Data[i]] * g_fp16ToFp32Manager.dict[input1Data[i]]);
            }
            input0Data += inner;
        }
    } else {
        float *input0Data = (float *)input0.cpuData;
        float *input1Data = (float *)input1.cpuData;
        for (int j = 0; j < round; j++) {
            for (int i = 0; i < len; i++) {
                input0Data[i] *= input1Data[i];
            }
            input0Data += inner;
        }
    }
}

MultiThreadAddToFloatOp::MultiThreadAddToFloatOp(float *input, float *output, int len, float alpha) {
    this->input = input;
    this->output = output;
    this->len = len;
    this->alpha = alpha;
}

void MultiThreadAddToFloatOp::Run() {
    for (int i = 0; i < len; i++) {
        output[i] += input[i] * alpha;
    }
}

static void RunMultiThreadAddToFloat(float *output, float *input, float alpha, int len, AliveThreadPool *pool) {
    if (len < 256 * 1024) {
        (MultiThreadAddToFloatOp(output, input, alpha, len)).Run();
        return;
    }
    int threadNum = pool->threads.size();
    int per = len / pool->threads.size();
    int cur = 0;
    std::vector<MultiThreadAddToFloatOp *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
        ops.push_back(new MultiThreadAddToFloatOp(output + cur, input + cur, alpha, end - cur));
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

void CpuAddToOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

    AssertInFastLLM(input0.dataType == DataType::FLOAT32 || input0.dataType == DataType::FLOAT16,
                    "AddTo error: Data's type should be float32 or float16.\n");
    AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

    int len = input0.Count(0);

    if (input0.dataType == DataType::FLOAT32) {
        float *input0Data = (float *)input0.cpuData;
        float *input1Data = (float *)input1.cpuData;
        RunMultiThreadAddToFloat(input0Data, input1Data, alpha, len, GetAlivePool());
    } else if (input0.dataType == DataType::FLOAT16) {
        uint16_t *input0Data = (uint16_t *)input0.cpuData;
        uint16_t *input1Data = (uint16_t *)input1.cpuData;
        for (int i = 0; i < len; i++) {
            input0Data[i] = float_to_half(g_fp16ToFp32Manager.dict[input0Data[i]] + g_fp16ToFp32Manager.dict[input1Data[i]] * alpha);
        }
    }
}

void CpuAttentionMaskOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &mask = *(datas.find("mask")->second);
    float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;

    int n = input.dims[0];
    int m = input.dims[1];
    int spatial = input.Count(2);

    AssertInFastLLM(mask.dataType == DataType::FLOAT32 || mask.dataType == input.dataType, "AttentionMask: mask's datatype should be float32.");
    if (input.dataType == DataType::FLOAT32 && mask.dataType == DataType::FLOAT32) {
        float *attnData = (float *)input.cpuData;
        float *maskData = (float *)mask.cpuData;
        for (int on = 0; on < n; on++) {
            for (int om = 0; om < m; om++) {
                for (int i = 0; i < spatial; i++) {
                    if (maskData[on * spatial + i] > 0.99) {
                        attnData[(on * m + om) * spatial + i] = maskValue;
                    }
                }
            }
        }
    } else if (input.dataType == DataType::FLOAT16 && mask.dataType == DataType::FLOAT32) {
        uint16_t *attnData = (uint16_t *)input.cpuData;
        float *maskData = (float *)mask.cpuData;
        for (int on = 0; on < n; on++) {
            for (int om = 0; om < m; om++) {
                for (int i = 0; i < spatial; i++) {
                    if (maskData[on * spatial + i] > 0.99) {
                        attnData[(on * m + om) * spatial + i] = float_to_half(maskValue);
                    }
                }
            }
        }
    } else if (input.dataType == DataType::FLOAT16 && mask.dataType == DataType::FLOAT16) {
        std::vector<float> floatMaskData;
        floatMaskData.resize(mask.Count(0));
        Float16ToFloat32((uint16_t *)mask.cpuData, floatMaskData.data(), mask.Count(0));
        uint16_t *attnData = (uint16_t *)input.cpuData;
        for (int on = 0; on < n; on++) {
            for (int om = 0; om < m; om++) {
                for (int i = 0; i < spatial; i++) {
                    if (floatMaskData[on * spatial + i] > 0.99) {
                        attnData[(on * m + om) * spatial + i] = float_to_half(maskValue);
                    }
                }
            }
        }
    } else {
        AssertInFastLLM(false, "AttentionMask error: Data's type should be float32 or float16.\n");
    }
}

void CpuAttentionExtendedMaskOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &mask = *(datas.find("mask")->second);
    int spatial = input.dims[3], n = input.dims[0], m = input.dims[1] * input.dims[2];

    AssertInFastLLM(mask.dataType == DataType::FLOAT32, "AttentionExtendedMask: mask's datatype should be float32.");
    if (input.dataType == DataType::FLOAT32) {
        float *maskData = (float *)mask.cpuData;
        float *attnData = (float *)input.cpuData;
        for (int on = 0; on < n; on++) {
            for (int om = 0; om < m; om++) {
                int o = on * m + om;
                for (int i = 0; i < spatial; i++) {
                    attnData[o * spatial + i] += maskData[on * spatial + i];
                }
            }
        }
    } else {
        ErrorInFastLLM("AttentionExtendedMask error: unsupport input's dataType.\n");
    }
}

void CpuTopKOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;

    AssertInFastLLM(input.dataType == DataType::FLOAT32, "TopK error: Data's type should be float32.\n");

    int dimsLen = input.dims.size();
    std::vector<int> dims = input.dims;
    dims[dimsLen - 1] = topk * 2;

    output.dataType = input.dataType;
    output.Resize(dims);
}

void CpuTopKOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;

    AssertInFastLLM(input.dataType == DataType::FLOAT32, "TopK error: Data's type should be float32.\n");

    float *inputData = (float *)input.cpuData;
    float *outputData = (float *)output.cpuData;

    int dimlens = input.dims.size();
    int outer = input.Count(0) / input.Count(dimlens - 1);
    int channel = input.dims[dimlens - 1];

    if (topk == 1) {
        for (int o = 0; o < outer; o++) {
            float maxValue = -1e100;
            int idx = -1;
            for (int i = 0; i < channel; i++) {
                if (inputData[i] > maxValue) {
                    maxValue = inputData[o * channel + i];
                    idx = i;
                }
            }
            outputData[o * 2] = idx;
            outputData[o * 2 + 1] = maxValue;
            inputData = inputData + channel;
        }
    } else {
        for (int o = 0; o < outer; o++) {
            std::set<std::pair<float, int>> vec;
            for (int i = 0; i < channel; i++) {
                if (vec.size() == topk && vec.begin()->first < inputData[i]) {
                    vec.erase(vec.begin());
                    vec.insert(std::make_pair(inputData[i], i));
                } else {
                    vec.insert(std::make_pair(inputData[i], i));
                }
            }

            int j = topk - 1;
            for (auto &t : vec) {
                outputData[o * topk * 2 + j * 2] = t.second;
                outputData[o * topk * 2 + j * 2 + 1] = t.first;
                j--;
            }

            inputData = inputData + channel;
        }
    }
}

void CpuPermuteOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &axisData = *(datas.find("axis")->second);
    std::vector<int> axis;
    for (int i = 0; i < axisData.Count(0); i++) {
        axis.push_back(((int32_t *)axisData.cpuData)[i]);
    }

    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Permute error: datatype should be float32 or float16.");
    AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");
    std::vector<int> new_dims;
    for (int i = 0; i < axis.size(); i++) {
        new_dims.push_back(input.dims[axis[i]]);
    }

    output.dataType = input.dataType;
    output.Resize(new_dims);
}

void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
    if (n < 4 || m < 4) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }
        return;
    }

#ifdef __aarch64__
    float32x4x2_t q01 = vtrnq_f32(vld1q_f32(pSrc), vld1q_f32(pSrc + srcStride));
    float32x4x2_t q23 = vtrnq_f32(vld1q_f32(pSrc + 2 * srcStride), vld1q_f32(pSrc + 3 * srcStride));

    float32x4_t qq0 = q01.val[0];
    float32x2_t d00 = vget_low_f32(qq0);
    float32x2_t d01 = vget_high_f32(qq0);

    float32x4_t qq1 = q01.val[1];
    float32x2_t d10 = vget_low_f32(qq1);
    float32x2_t d11 = vget_high_f32(qq1);

    float32x4_t qq2 = q23.val[0];
    float32x2_t d20 = vget_low_f32(qq2);
    float32x2_t d21 = vget_high_f32(qq2);

    float32x4_t qq3 = q23.val[1];
    float32x2_t d30 = vget_low_f32(qq3);
    float32x2_t d31 = vget_high_f32(qq3);

    vst1q_f32(pDst, vcombine_f32(d00, d20));
    vst1q_f32(pDst + 1 * dstStride, vcombine_f32(d10, d30));
    vst1q_f32(pDst + 2 * dstStride, vcombine_f32(d01, d21));
    vst1q_f32(pDst + 3 * dstStride, vcombine_f32(d11, d31));
#else
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            pDst[j * dstStride + i] = pSrc[i * srcStride + j];
        }
    }
#endif
}

void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
    int per = 4;

    for (int i = 0; i < n; i += per) {
        for (int j = 0; j < m; j += per) {
            Transpose4x4(pDst + j * dstStride + i, pSrc + i * srcStride + j, dstStride, srcStride, std::min(per, n - i), std::min(per, m - j));
        }
    }
}

MultiThreadTransposeOp::MultiThreadTransposeOp(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
    this->pDst = pDst;
    this->pSrc = pSrc;
    this->n = n;
    this->m = m;
    this->dstStride = dstStride;
    this->srcStride = srcStride;
}

void MultiThreadTransposeOp::Run() { Transpose(pDst, pSrc, dstStride, srcStride, n, m); }

MultiThreadSiluOp::MultiThreadSiluOp(float *input, int len, float *output, int n, int inputStride, int outputStride) {
    this->input = input;
    this->output = output;
    this->n = n;
    this->len = len;
    this->inputStride = inputStride;
    this->outputStride = outputStride;
    this->mid = mid;
}

void MultiThreadSiluOp::Run() {
    for (int o = 0; o < n; o++) {
        float *cur = (float *)input + o * inputStride;
        float *out = (float *)output + o * outputStride;

        int i = 0;
#ifdef __aarch64__
        float32x4_t c1 = vdupq_n_f32(1.0f);
        for (; i + 3 < len; i += 4) {
            float32x4_t vx = vld1q_f32(cur + i);
            float32x4_t vdiv = vaddq_f32(c1, exp_ps(vnegq_f32(vx)));
            vx = vdivq_f32(vx, vdiv);
            vst1q_f32(out + i, vx);
        }
#endif
        for (; i < len; i++) {
            float x = cur[i];
            out[i] = x / (1.0 + expf(-x));
        }
    }
}

void SiluMultiThread(float *input, int len, float *output, int n, int inputStride, int outputStride, AliveThreadPool *pool) {
    int threadNum = pool->threads.size();
    int per = len / threadNum;
    int cur = 0;
    std::vector<MultiThreadSiluOp *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
        ops.push_back(new MultiThreadSiluOp(input + cur, end - cur, output + cur, n, inputStride, outputStride));
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

float gelu(float x) { return x * 0.5f * (1.0f + erf(x / sqrt(2.0))); }