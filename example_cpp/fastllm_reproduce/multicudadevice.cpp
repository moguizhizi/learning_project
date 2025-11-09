#include "multicudadevice.h"

MultiCudaDevice::MultiCudaDevice(CudaDevice *cudaDevice) {
    this->cudaDevice = cudaDevice;
    this->deviceType = "multicuda";

    this->ops["MLP"] = (BaseOperator *)(new MultiCudaMLPOp());
    this->ops["Linear"] = (BaseOperator *)(new MultiCudaLinearOp());
    this->ops["MergeMOE"] = (BaseOperator *)(new MultiCudaMergeMOE());
    this->ops["MergeAttention"] = (BaseOperator *)(new MultiCudaMergeAttention());
}