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

// float的input, int8的weight, 直接计算得到float的output
void Int8LinearPart(
    float *inputData, uint8_t *weightData, float *biasData, float *outputData, LowBitConfig *configs, int n, int m, int k, int st, int end);

// float的input, int4g的weight, 直接计算得到float的output
void Int4GroupLinearPart(float *inputData,
                         uint8_t *weightData,
                         float *biasData,
                         float *outputData,
                         LowBitConfig *configs,
                         int n,
                         int m,
                         int k,
                         int st,
                         int end,
                         int group,
                         int groupCnt);

// float的input, int4的weight, 直接计算得到float的output
void Int4LinearPart(
    float *inputData, uint8_t *weightData, float *biasData, float *outputData, LowBitConfig *configs, int n, int m, int k, int st, int end);

void GetArrayMinMax(float *a, int len, float &minValue, float &maxValue);

void QuantizationAll(float *fValue, uint8_t *uValue, int len, LowBitConfig *config);
