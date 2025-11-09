#include "multicudadevice.h"

#include "fastllm-cuda.cuh"

MultiCudaDevice::MultiCudaDevice(CudaDevice *cudaDevice) {
    this->cudaDevice = cudaDevice;
    this->deviceType = "multicuda";

    this->ops["MLP"] = (BaseOperator *)(new MultiCudaMLPOp());
    this->ops["Linear"] = (BaseOperator *)(new MultiCudaLinearOp());
    this->ops["MergeMOE"] = (BaseOperator *)(new MultiCudaMergeMOE());
    this->ops["MergeAttention"] = (BaseOperator *)(new MultiCudaMergeAttention());
}

bool MultiCudaDevice::Malloc(void **ret, size_t size) {
    *ret = FastllmCudaMalloc(size);
    return true;
}

bool MultiCudaDevice::Free(void *ret) {
    FastllmCudaFree(ret);
    return true;
}

bool MultiCudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
    FastllmCudaCopyFromHostToDevice(dst, src, size);
    return true;
}

bool MultiCudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
    FastllmCudaCopyFromDeviceToHost(dst, src, size);
    return true;
}

bool MultiCudaDevice::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (this->ops.find(opType) == this->ops.end()) {
        if (((BaseDevice *)this->cudaDevice)->ops.find(opType) == ((BaseDevice *)this->cudaDevice)->ops.end()) {
            return false;
        } else {
            return ((BaseDevice *)this->cudaDevice)->CanRun(opType, datas, floatParams, intParams);
        }
    } else {
        return this->ops[opType]->CanRun(opType, datas, floatParams, intParams);
    }
}

// 对某一个算子进行形状推理
void MultiCudaDevice::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (this->ops.find(opType) == this->ops.end()) {
        ((BaseDevice *)this->cudaDevice)->Reshape(opType, datas, floatParams, intParams);
    } else {
        this->ops[opType]->Reshape(opType, datas, floatParams, intParams);
    }
}

// 对某一个算子进行推理
void MultiCudaDevice::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (this->ops.find(opType) == this->ops.end()) {
        ((BaseDevice *)this->cudaDevice)->Run(opType, datas, floatParams, intParams);
    } else {
        this->ops[opType]->Run(opType, datas, floatParams, intParams);
    }
}