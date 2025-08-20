#include "cpudevice.h"
#include "basellm.h"
#include "fastllm.h"
#include "file_utils.hpp"
#include "utils.h"
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