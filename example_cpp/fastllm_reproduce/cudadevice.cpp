#include "cudadevice.h"
#include "fastllm-cuda.cuh"
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