#pragma once // 或者用 #ifndef 方式

#include "alivethreadpool.h"
#include "device.h"

class CpuToFloat16 : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuToFloat32 : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuConvertToFloat16 : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuConvertToFloat32 : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAttention : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

  protected:
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuCopyKVCacheOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuEmbedding : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuLayerNormOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuRMSNormOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

void RunMultiThreadRMSNormFloat(float *output, float *input, float *weight, int outer, int channels, float eps, AliveThreadPool *pool);

class CpuConv2DOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

  protected:
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuLinearOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

  protected:
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

void DoCpuLinearReshape(Data &input, Data &weight, Data &output);
