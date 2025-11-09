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