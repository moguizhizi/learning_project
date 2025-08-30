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
void DoCpuLinear(Data &input, Data &weight, const Data &bias, Data &output);

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

struct MultiThreadLinearInt4Op : MultiThreadBaseOp {
    uint8_t *a;
    uint8_t *b;
    int32_t *c;
    int n, m, k, kstride;
    int *weightSums, *weightZeros;
    float *scales, *bias;
    LowBitConfig *config;
    int *inputSums;

    MultiThreadLinearInt4Op(uint8_t *a,
                            uint8_t *b,
                            int32_t *c,
                            int n,
                            int m,
                            int k,
                            int kstride,
                            int *weightSums,
                            int *weightZeros,
                            float *scales,
                            float *bias,
                            LowBitConfig *config,
                            int *inputSums);

    void Run();
};

// a = [n, m], b = [k, m], c = aT(b') = [n, k]
void MultiplyInt4MultiThread(uint8_t *a,
                             uint8_t *b,
                             int32_t *c,
                             int n,
                             int m,
                             int k,
                             int *weightSums,
                             int *weightZeros,
                             float *scales,
                             float *bias,
                             std::vector<LowBitConfig> &configs,
                             int threadNum);

void GetArrayMinMax(float *a, int len, float &minValue, float &maxValue);

void QuantizationAll(float *fValue, uint8_t *uValue, int len, LowBitConfig *config);

struct MultiThreadOnlineQuantizationOp : MultiThreadBaseOp {
    float *input;
    uint8_t *output;
    LowBitConfig *configs;
    int n, m, group, groupCnt;
    float *inputSums, *iscales, *izeros;
    int permuteType;

    MultiThreadOnlineQuantizationOp(float *input,
                                    uint8_t *output,
                                    LowBitConfig *configs,
                                    int n,
                                    int m,
                                    int group,
                                    int groupCnt,
                                    float *inputSums,
                                    float *iscales,
                                    float *izeros,
                                    int permuteType);

    void Run();
};

void OnlineQuantization(float *inputData,
                        std::vector<uint8_t> &uinput,
                        std::vector<LowBitConfig> &inputConfigs,
                        int n,
                        int m,
                        int group,
                        int groupCnt,
                        std::vector<float> &inputSums,
                        std::vector<float> &iscales,
                        std::vector<float> &izeros,
                        int permuteType);

struct MultiThreadLinearInt4NoZeroOp : MultiThreadBaseOp {
    uint8_t *a, *b;
    int32_t *c;
    int n, m, k, kstride;
    int *weightSums;
    float *weightMins, *scales, *bias;
    LowBitConfig *config;
    float *inputSums;

    MultiThreadLinearInt4NoZeroOp(uint8_t *a,
                                  uint8_t *b,
                                  int32_t *c,
                                  int n,
                                  int m,
                                  int k,
                                  int kstride,
                                  int *weightSums,
                                  float *weightMins,
                                  float *scales,
                                  float *bias,
                                  LowBitConfig *config,
                                  float *inputSums);

    void Run();
};

void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a,
                                        uint8_t *b,
                                        float *c,
                                        int n,
                                        int m,
                                        int k,
                                        int *weightSums,
                                        float *weightMins,
                                        float *scales,
                                        float *bias,
                                        std::vector<float> &inputSums,
                                        std::vector<float> &iscales,
                                        std::vector<float> &izeros,
                                        std::vector<LowBitConfig> &configs,
                                        int startTid,
                                        int threadNum,
                                        int group,
                                        int groupCnt,
                                        std::vector<MultiThreadBaseOp *> &ops,
                                        AliveThreadPool *pool);
