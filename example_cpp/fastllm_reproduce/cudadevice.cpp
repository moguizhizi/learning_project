#include "cudadevice.h"
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "file_utils.hpp"

CudaDevice::CudaDevice() {}

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

void DoCudaAttentionReshape(Data &q, Data &v, Data &output) {
    std::vector<int> dims = {q.dims[0], q.dims[1], v.dims[2]};
    output.dataType = q.dataType;
    output.Resize(dims);
}

void DoCudaLinearReshape(Data &input, Data &weight, Data &output) {
    weight.weightType = WeightType::LINEAR;
    std::vector<int> dims = input.dims;
    dims.back() = weight.dims[0];

    output.dataType = input.dataType;
    output.Resize(dims);
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
    if (input->expansionDims.size() != 0)
        output->Expansion(input->expansionDims);
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
    if (input->expansionDims.size() != 0)
        output->Expansion(input->expansionDims);
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
    AssertInFastLLM(q.dataType == DataType::FLOAT32 || q.dataType == DataType::FLOAT16, "Attention's input's type should be float32 or float16.\n");

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
    AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                    "Mul error: Data's type should be float32 or float16.\n");
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

void CudaNearlyRotatePosition2DOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
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
