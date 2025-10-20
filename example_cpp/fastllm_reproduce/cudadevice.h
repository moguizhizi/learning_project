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

void DoCudaAttentionReshape(Data &q, Data &v, Data &output);
void DoCudaLinearReshape(Data &input, Data &weight, Data &output);

class CudaLinearOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaToFloat16 : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaToFloat32 : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaConvertToFloat16 : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaConvertToFloat32 : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaAttention : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaEmbedding : BaseOperator {
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaMulToOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaAddToOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaMulOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaApplyLognAttnOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaLlamaRotatePosition2DOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaNearlyRotatePosition2DOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaRotatePosition2DOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaSplitBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaCatBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaMulBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};