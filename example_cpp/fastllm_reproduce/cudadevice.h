#pragma once // 或者用 #ifndef 方式

#include "basellm.h"
#include "device.h"

class CudaDevice : BaseDevice {
  public:
    CudaDevice();

    bool Malloc(void **ret, size_t size); // 分配尺寸为size的空间
    bool Free(void *ret);                 // 释放ret

    bool CopyDataToCPU(void *dst, void *src, size_t size);
    bool CopyDataFromCPU(void *dst, void *src, size_t size);
};

void DoCudaLinearReshape(Data &input, Data &weight, Data &output);

class CudaLinearOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaToFloat16 : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};