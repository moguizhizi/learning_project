#include "cudadevice.h"

#include <numeric>

#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "file_utils.hpp"

constexpr float EPSILON = 1e-9f;     // For numerical safety
constexpr int SharedExpertIndex = 0; // Reserved expert ID

CudaDevice::CudaDevice() {
    this->deviceType = "cuda";
    this->ops["ToFloat16"] = (BaseOperator *)(new CudaToFloat16());
    this->ops["ToFloat32"] = (BaseOperator *)(new CudaToFloat32());
    this->ops["ConvertToFloat16"] = (BaseOperator *)(new CudaConvertToFloat16());
    this->ops["ConvertToFloat32"] = (BaseOperator *)(new CudaConvertToFloat32());

    this->ops["Attention"] = (BaseOperator *)(new CudaAttention());
    // this->ops["MergeAttention"] = (BaseOperator*)(new CudaMergeAttention());
    this->ops["Embedding"] = (BaseOperator *)(new CudaEmbedding());
    this->ops["Linear"] = (BaseOperator *)(new CudaLinearOp());
    this->ops["Repeat"] = (BaseOperator *)(new CudaRepeatOp());
    this->ops["MatMulTransB"] = (BaseOperator *)(new CudaMatMulTransBOp());
    this->ops["Relu"] = (BaseOperator *)(new CudaReluOp());
    this->ops["Mul"] = (BaseOperator *)(new CudaMulOp());
    this->ops["AddTo"] = (BaseOperator *)(new CudaAddToOp());
    this->ops["MulTo"] = (BaseOperator *)(new CudaMulToOp());
    this->ops["AlibiMask"] = (BaseOperator *)(new CudaAlibiMaskOp());
    this->ops["RotatePosition2D"] = (BaseOperator *)(new CudaRotatePosition2DOp());
    this->ops["NearlyRotatePosition2D"] = (BaseOperator *)(new CudaNearlyRotatePosition2DOp());
    this->ops["LlamaRotatePosition2D"] = (BaseOperator *)(new CudaLlamaRotatePosition2DOp());
    this->ops["ApplyLognAttn"] = (BaseOperator *)(new CudaApplyLognAttnOp());

    this->ops["SplitBatch"] = (BaseOperator *)(new CudaSplitBatchOp());
    this->ops["CatBatch"] = (BaseOperator *)(new CudaCatBatchOp());
    this->ops["MulBatch"] = (BaseOperator *)(new CudaMulBatchOp());
    this->ops["MatMulTransBBatch"] = (BaseOperator *)(new CudaMatMulTransBBatchOp());
}

bool CudaDevice::Malloc(void **ret, size_t size) {
    *ret = FastllmCudaMalloc(size);
    return true;
}

bool CudaDevice::Free(void *ret) {
    FastllmCudaFree(ret);
    return true;
}

bool CudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
    FastllmCudaCopyFromHostToDevice(dst, src, size);
    return true;
}

bool CudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
    FastllmCudaCopyFromDeviceToHost(dst, src, size);
    return true;
}

void DoCudaLinearReshape(Data &input, Data &weight, Data &output) {
    weight.weightType = WeightType::LINEAR;
    std::vector<int> dims = input.dims;
    dims.back() = weight.dims[0];

    output.dataType = input.dataType;
    output.Resize(dims);
}

void DoCudaLinear(Data &input, Data &weight, const Data &bias, Data &output) {
    output.Allocate();
    int n = input.Count(0) / input.dims.back();
    int m = input.dims.back();
    int k = output.dims.back();
    if (bias.dataType != DataType::FLOAT32) {
        ErrorInFastLLM("Linear error: unsupport bias' dataType.\n");
    } else if (input.dataType == DataType::FLOAT16) {
        if (weight.dataType == DataType::FLOAT32) {
            FastllmCudaHalfMatMulFloat32(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::FLOAT16) {
            FastllmCudaHalfMatMulFloat16(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT8) {
            FastllmCudaHalfMatMulFloatInt8(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT4_GROUP) {
            FastllmCudaHalfMatMulFloatInt4Group(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT4_NOZERO) {
            FastllmCudaHalfMatMulFloatInt4NoZero(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::FP8_E4M3) {
            FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, output, n, m, k);
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
    } else if (input.dataType == DataType::FLOAT32) {
        if (weight.dataType == DataType::FLOAT32) {
            FastllmCudaMatMulFloat32(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::FLOAT16) {
            FastllmCudaMatMulFloat16(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT8) {
            FastllmCudaMatMulFloatInt8(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT4) {
            FastllmCudaMatMulFloatInt4(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT4_NOZERO) {
            FastllmCudaMatMulFloatInt4NoZero(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT4_GROUP) {
            FastllmCudaMatMulFloatInt4Group(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::FP8_E4M3) {
            FastllmCudaMatMulFloatFP8E4M3(input, weight, bias, output, n, m, k);
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
    } else {
        ErrorInFastLLM("Linear error: unsupport input's dataType.\n");
    }
}

void DoCudaSplitReshape(Data &input, int axis, int start, int end, Data &output) {
    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    start = std::max(0, std::min(input.dims[axis] - 1, start));
    end = std::max(0, std::min(input.dims[axis], end));
    std::vector<int> dims = input.dims;
    dims[axis] = end - start;

    output.dataType = input.dataType;
    output.Resize(dims);
}

void DoCudaSplit(Data &input, int axis, int start, int end, Data &output) {
    output.Allocate();

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    start = std::max(0, std::min(input.dims[axis] - 1, start));
    end = std::max(0, std::min(input.dims[axis], end));

    int outer = input.Count(0) / input.dims[axis];
    int outputStride = output.Count(axis);
    int inputStride = input.Count(axis);
    int uniteSize = output.unitSize;
    int inner = input.strides[axis];

    FastllmCudaMemcpy2DDeviceToDevice((uint8_t *)output.cudaData, outputStride * uniteSize,
        (uint8_t *)input.cudaData + start * inner * uniteSize, inputStride * uniteSize, (end - start) * inner * uniteSize, outer);
}

void DoCudaPermuteSelf(Data &input, const std::vector<int> &axis) {
    bool same = false;
    same |= ((axis == std::vector<int>{1, 2, 0} || axis == std::vector<int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
    same |= ((axis == std::vector<int>{2, 0, 1, 3}) && input.dims[2] == 1);
    same |= ((axis == std::vector<int>{2, 0, 1, 3}) && input.dims[0] == 1 && input.dims[1] == 1);
    same |= ((axis == std::vector<int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
    same |= ((axis == std::vector<int>{1, 0, 2, 3}) && (input.dims[0] == 1 || input.dims[1] == 1));
    same |= ((axis == std::vector<int>{1, 2, 0, 3}) && input.dims[1] == 1 && input.dims[2] == 1);
    if (same) {
        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }
        input.Resize(new_dims);
        return;
    }

    FastllmCudaPermute(input, axis);
}

void DoCudaCatDirect(Data &input0, Data &input1, int axis) {
    int dimsLen = input1.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    if (input0.dims.size() == 0) {
        input0.Resize(input1.dims);
        input0.Allocate();

        int outer = input1.Count(0) / input1.Count(axis);
        int inputStride0 = input0.Count(axis);
        int inputStride1 = input1.Count(axis);
        int inner = input1.strides[axis];
        int unitSize = input0.unitSize;

        FastllmCudaMemcpy2DDeviceToDevice((uint8_t *)input0.cudaData, inputStride0 * unitSize, (uint8_t *)input1.cudaData,
            inputStride1 * unitSize, inputStride1 * unitSize, outer);

    } else {
        std::vector<int> dims = input0.dims;
        std::vector<int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        FastllmCudaMemcpy2DDeviceToDevice((uint8_t *)input0.cudaData + oldDims[axis] * inner * unitSize, input0Stride * unitSize,
            (uint8_t *)input1.cudaData, input1Stride * unitSize, input1.dims[axis] * inner * unitSize, outer);
    }
}

void DoCudaAttentionReshape(Data &q, Data &v, Data &output) {
    std::vector<int> dims = {q.dims[0], q.dims[1], v.dims[2]};
    output.dataType = q.dataType;
    output.Resize(dims);
}

void DoCudaAttention(Data &q, Data &k, Data &v, Data &mask, Data &output, int group, float scale, int maskType) {
    output.Allocate();
    if (q.dataType == DataType::FLOAT32) {
        FastllmCudaAttention(q, k, v, mask, output, group, scale, maskType);
    } else if (q.dataType == DataType::FLOAT16) {
#ifdef CUDA_NO_TENSOR_CORE
        Data q32, k32, v32, mask32, output32;
        ToDataType(q, q32, DataType::FLOAT32);
        ToDataType(k, k32, DataType::FLOAT32);
        ToDataType(v, v32, DataType::FLOAT32);
        ToDataType(output, output32, DataType::FLOAT32);
        if (mask.dims.size() > 0) ToDataType(mask, mask32, DataType::FLOAT32);
        FastllmCudaAttention(q32, k32, v32, mask32, output32, group, scale, maskType);
        ToDataType(output32, output, DataType::FLOAT16);
#else
        FastllmCudaHalfAttention(q, k, v, mask, output, group, scale, maskType);
#endif
    }
}

void DoCudaSwigluReshape(Data &input, Data &output) {
    std::vector<int> dims = input.dims;
    dims[dims.size() - 1] /= 2;
    output.dataType = input.dataType;
    output.Resize(dims);
}

void DoCudaSwiglu(Data &input, Data &output) {
    output.Allocate();
    FastllmCudaSwiglu(input, output);
}

/**
 * @brief Select top-k experts based on router logits (optionally with bias).
 */
std::vector<std::pair<float, int>> ComputeRouterScores(const float *logits, const float *bias, int m) {
    std::vector<std::pair<float, int>> routerScores;
    routerScores.reserve(m);
    for (int i = 0; i < m; i++) {
        float score = -logits[i];
        if (bias) score -= bias[i];
        routerScores.emplace_back(score, i);
    }

    return routerScores;
}

/**
 * @brief Partially sort and pick top-k expert indices.
 */
std::vector<int> SelectTopExperts(std::vector<std::pair<float, int>> &routerScores, int topk) {
    std::vector<int> selectedExperts;
    selectedExperts.reserve(topk);

    std::partial_sort(routerScores.begin(), routerScores.begin() + topk, routerScores.end());

    for (auto &router : routerScores) {
        selectedExperts.push_back(router.second);
    }

    return selectedExperts;
}

/**
 * @brief Normalize routing weights for selected experts.
 */
std::vector<ExpertRoute> NormalizeExpertWeights(const float *logits, const std::vector<int> &selectedExperts, float routeScale, bool needNorm) {
    float sum = 0.0f;
    if (needNorm) {
        for (int i : selectedExperts) {
            sum += logits[i];
        }
        sum = std::max(sum, EPSILON);
    } else {
        sum = 1.0f;
    }

    std::vector<ExpertRoute> routes;
    routes.reserve(selectedExperts.size() + 1);
    for (int i : selectedExperts) {
        routes.push_back({i + 1, logits[i] / sum * routeScale});
    }

    return routes;
}

std::vector<ExpertRoute> RouteMoE(const float *logits, const float *bias, int m, int topk, float routeScale, bool needNorm, float *sharedScale) {
    auto scores = ComputeRouterScores(logits, bias, m);
    auto selectedExperts = SelectTopExperts(scores, topk);
    auto routedExperts = NormalizeExpertWeights(logits, selectedExperts, routeScale, needNorm);

    if (sharedScale != nullptr) {
        routedExperts.push_back({SharedExpertIndex, *sharedScale});
    } else {
        float shareWeight =
            std::accumulate(routedExperts.begin(), routedExperts.end(), 0.0f, [](float acc, ExpertRoute &r) { return acc + r.weight; });
        routedExperts.push_back({SharedExpertIndex, shareWeight});
    }

    return routedExperts;
}

void DoCudaMergeMOE(Data &input, Data &output, Data &gateBias, Data &logits, Data &w1, Data &w2, Data &w3, Data **weights, Data **biass,
    int topk, int needNorm, float sharedScale, float routeScale) {
    output.Allocate();

    int batch = input.dims[0];
    float *cpuRouterLogits = (float *)logits.cpuData;
    int m = logits.dims.back();

    Data &bias = gateBias;
    Data curInput, curOutput;
    curInput.ToDevice(input.dataDevice);
    curOutput.ToDevice(output.dataDevice);
    curOutput.dataType = output.dataType;

    float *cpuData = nullptr;
    if (bias.dims.size() > 0) {
        ToDataType(bias, DataType::FLOAT32);
        bias.ToDevice(DataDevice::CPU);
        cpuData = (float *)bias.cpuData;
    }

    for (int b = 0; b < batch; b++) {
        float *curLogits = cpuRouterLogits + b * m;

        auto routedExperts = RouteMoE(curLogits, cpuData, m, topk, routeScale, needNorm, &sharedScale);

        DoCudaSplitReshape(input, 0, b, b + 1, curInput);
        DoCudaSplit(input, 0, b, b + 1, curInput);

        curOutput.Resize(curInput.dims);
        curOutput.Allocate(0.0f);

        bool first = true;
        for (ExpertRoute expert : routedExperts) {
            int expertIndex = expert.expertIndex;
            float expertWeight = expert.weight;

            // 没有权重 → 该专家无效
            if (weights[2 * expertIndex] == nullptr) {
                continue;
            }

            // Expert MLP Forward: Linear → SwiGLU → Linear
            DoCudaLinearReshape(curInput, *weights[2 * expertIndex], w3);
            DoCudaLinear(curInput, *weights[2 * expertIndex], Data(), w3);

            DoCudaSwigluReshape(w3, w1);
            DoCudaSwiglu(w3, w1);

            DoCudaLinearReshape(w1, *weights[2 * expertIndex + 1], w2);
            DoCudaLinear(w1, *weights[2 * expertIndex + 1], Data(), w2);

            FastllmCudaAddTo(curOutput, w2, expertWeight);
            FastllmCudaCopyFromDeviceToDevice(
                (uint8_t *)output.cudaData + b * curOutput.GetBytes(), (uint8_t *)curOutput.cudaData, curOutput.GetBytes());
        }
    }
}

void CudaLinearOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);

    AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
    AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");
    DoCudaLinearReshape(input, weight, output);
}

bool CudaLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (intParams.find("exType") != intParams.end()) {
        return false;
    }
    return true;
}

void CudaToFloat16::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);

    if (input.dataType == DataType::FLOAT16) {
        return;
    }

    if (input.dims.size() == 0) {
        input.dataType = DataType::FLOAT16;
        input.UpdateUnitSize();
        return;
    }

    if (input.dataType == DataType::FLOAT32) {
        float *old = (float *)input.cudaData;
        int len = input.Count(0);
        input.dataType = DataType::FLOAT16;
        input.UpdateUnitSize();
        input.cudaData = FastllmCudaMalloc(input.GetBytes());
        FastllmFloatToHalf(old, input.cudaData, len);
        FastllmCudaFree(old);
    } else {
        AssertInFastLLM(false, "CudaToFloat16 only support float32 to float16.\n");
    }
}

void CudaToFloat32::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);

    if (input.dataType == DataType::FLOAT32) {
        return;
    }

    if (input.dims.size() == 0) {
        input.dataType = DataType::FLOAT32;
        input.UpdateUnitSize();
        return;
    }

    if (input.dataType == DataType::FLOAT16) {
        uint16_t *old = (uint16_t *)input.cudaData;
        int len = input.Count(0);
        input.dataType = DataType::FLOAT32;
        input.UpdateUnitSize();
        input.cudaData = FastllmCudaMalloc(input.GetBytes());
        FastllmHalfToFloat(old, input.cudaData, len);
        FastllmCudaFree(old);
    } else {
        AssertInFastLLM(false, "CudaToFloat32 only support float16 to float32.\n");
    }
}

void CudaConvertToFloat16::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data *input = (datas.find("input")->second);
    Data *output = (datas.find("output")->second);
    output->dataType = DataType::FLOAT16;
    output->Resize(input->dims);
    if (input->expansionDims.size() != 0) output->Expansion(input->expansionDims);
}

void CudaConvertToFloat16::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    if (input.dataType == DataType::FLOAT16) {
        FastllmCudaCopyFromDeviceToDevice(output.cudaData, input.cudaData, input.GetBytes());
        return;
    }
    if (input.dataType == DataType::FLOAT32) {
        FastllmFloatToHalf(input.cudaData, output.cudaData, input.Count(0));
    } else {
        ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
    }
}

void CudaConvertToFloat32::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data *input = (datas.find("input")->second);
    Data *output = (datas.find("output")->second);
    output->dataType = DataType::FLOAT32;
    output->Resize(input->dims);
    if (input->expansionDims.size() != 0) output->Expansion(input->expansionDims);
}

void CudaConvertToFloat32::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    if (input.dataType == DataType::FLOAT32) {
        FastllmCudaCopyFromDeviceToDevice(output.cudaData, input.cudaData, input.GetBytes());
        return;
    }
    if (input.dataType == DataType::FLOAT16) {
        FastllmHalfToFloat(input.cudaData, output.cudaData, input.Count(0));
    } else {
        ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
    }
}

void CudaAttention::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &q = *(datas.find("q")->second);
    Data &k = *(datas.find("k")->second);
    Data &v = *(datas.find("v")->second);
    Data &output = *(datas.find("output")->second);
    int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];

    AssertInFastLLM(q.dims.size() == 3 && k.dims.size() == 3 && v.dims.size() == 3, "Attention: dims of q, k, v should be 3.\n");
    AssertInFastLLM(q.dims[2] == k.dims[2], "Attention: q.dims[2] should be equal to k.dims[2].\n");
    AssertInFastLLM(k.dims[1] == v.dims[1], "Attention: k.dims[1] should be equal to v.dims[1].\n");
    AssertInFastLLM(k.dims[0] == v.dims[0], "Attention: k.dims[0] should be equal to v.dims[0].\n");
    AssertInFastLLM(q.dims[0] == k.dims[0] * group, "Attention: q.dims[0] should be equal to k.dims[0] * group.\n");

    AssertInFastLLM(q.dataType == k.dataType && q.dataType == v.dataType, "Attention: q, k, v's datatype should be same.\n");
    AssertInFastLLM(
        q.dataType == DataType::FLOAT32 || q.dataType == DataType::FLOAT16, "Attention's input's type should be float32 or float16.\n");

    DoCudaAttentionReshape(q, v, output);
}

bool CudaEmbedding::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (GetLowMemMode() || !GetCudaEmbedding()) {
        return false;
    }
    Data &input = *(datas.find("input")->second);
    if (input.dataType != DataType::FLOAT32) {
        return false;
    }
    return true;
}

void CudaEmbedding::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);

    output.Allocate();

    FastllmCudaEmbedding(input, weight, output);
}

void CudaMulToOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

    AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
        "MulTo error: Data's type should be float32 or float16.\n");
    AssertInFastLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");
    FastllmCudaMulTo(input0, input1, alpha);
}

void CudaAddToOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input0 = *(datas.find("input0")->second);
    Data &input1 = *(datas.find("input1")->second);
    float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

    AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
        "AddTo error: Data's type should be float32 or float16.\n");
    AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");
    FastllmCudaAddTo(input0, input1, alpha);
}

void CudaMulOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();

    float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
    AssertInFastLLM(
        input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16, "Mul error: Data's type should be float32 or float16.\n");
    FastllmCudaMul(input, v, output);
}

void CudaApplyLognAttnOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &lognAttn = *(datas.find("lognAttn")->second);
    Data &positionIds = *(datas.find("positionIds")->second);

    FastllmCudaApplyLognAttn(input, lognAttn, positionIds);
}

void CudaLlamaRotatePosition2DOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &data = *(datas.find("input")->second);
    Data &positionIds = *(datas.find("positionIds")->second);
    Data &sinData = *(datas.find("sin")->second);
    Data &cosData = *(datas.find("cos")->second);
    int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

    FastllmCudaLlamaRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
}

void CudaNearlyRotatePosition2DOp::Run(
    const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &data = *(datas.find("input")->second);
    Data &positionIds = *(datas.find("positionIds")->second);
    Data &sinData = *(datas.find("sin")->second);
    Data &cosData = *(datas.find("cos")->second);
    int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;

    FastllmCudaNearlyRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
}

void CudaRotatePosition2DOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &data = *(datas.find("input")->second);
    Data &positionIds = *(datas.find("positionIds")->second);
    Data &sinData = *(datas.find("sin")->second);
    Data &cosData = *(datas.find("cos")->second);
    int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;

    FastllmCudaRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
}

void CudaMatMulTransBOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
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

void CudaMatMulTransBOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
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
    int k = input1.dims[input1.dims.size() - 2];
    int batch0 = input0.Count(0) / input0Spatial;
    int batch1 = input1.Count(0) / input1Spatial;

    int outputSpatial = output.Count(output.dims.size() - 2) * group;
    FastllmCudaBatchMatMulTransB(
        input0, input1, output, input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride, batch1, n, m, k, alpha);
}

void CudaConv2DOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &bias = *(datas.find("bias")->second);

    output.Allocate();

    int inputChannels = intParams.find("inputChannels")->second;
    int outputChannels = intParams.find("outputChannels")->second;
    int kernelH = intParams.find("kernelH")->second;
    int kernelW = intParams.find("kernelW")->second;
    int padH = intParams.find("padH")->second;
    int padW = intParams.find("padW")->second;
    int strideH = intParams.find("strideH")->second;
    int strideW = intParams.find("strideW")->second;

    std::vector<int> dims = input.dims;
    int inputHeight = dims[2], inputWidth = dims[3];
    int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
    int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;

    FastllmCudaConv2DFloat32(input, weight, bias, inputChannels, outputChannels, kernelH, kernelW, strideH, strideW, padH, padW, output);
}

void CudaRepeatOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;
    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;

    output.Allocate();

    int outer = output.Count(0) / output.Count(axis);
    int inputStride = input.Count(axis);
    int outputStride = output.Count(axis);
    int channels = input.dims[axis];
    int inner = input.strides[axis];
    int unitSize = input.unitSize;

    FastllmCudaRepeat(input.cudaData, output.cudaData, outer, repeatTimes, inputStride * unitSize, outputStride * unitSize,
        channels * inner * unitSize, channels * inner * unitSize);
}

void CudaReluOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    output.Allocate();
    AssertInFastLLM(input.dataType == DataType::FLOAT32, "Relu error: Data's type should be float32\n");
    FastllmCudaRelu(input, output);
}

void CudaAlibiMaskOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &mask = *(datas.find("mask")->second);
    float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
    FastllmCudaAlibiMask(input, mask, maskValue);
}

bool CudaLayerNormOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int inner = input.strides[axis];
    return inner == 1;
}

void CudaLayerNormOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &gamma = *(datas.find("gamma")->second);
    Data &beta = *(datas.find("beta")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    output.Allocate();
    FastllmCudaLayerNorm(input, gamma, beta, output, axis);
}

bool CudaRMSNormOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    return true;
}

void CudaRMSNormOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &output = *(datas.find("output")->second);

    AssertInFastLLM(
        input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16, "RMSNorm error: datatype should be float32 or float16.");

    output.Allocate();

    float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
    FastllmCudaRMSNorm(input, weight, output, eps);
}