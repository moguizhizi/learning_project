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
void DoCpuCatDirect(Data &input0, Data &input1, int axis);

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

class CpuSplitOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

struct MultiThreadSliceOp : MultiThreadBaseOp {
    uint8_t *input, *output;
    int outer, inputStride, outputStride, copyLen;

    MultiThreadSliceOp(uint8_t *output, uint8_t *input, int outer, int outputStride, int inputStride, int copyLen);

    void Run();
};

static void RunMultiThreadSlice(uint8_t *output, uint8_t *input, int outer, int inputStride, int outputStride, int copyLen, AliveThreadPool *pool);

class CpuRepeatOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuCatOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuCatDirectOp : BaseOperator {
  protected:
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

struct MultiThreadMatMulSingleOp : MultiThreadBaseOp {
    float *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;

    MultiThreadMatMulSingleOp(float *input0Base,
                              float *input1Base,
                              float *outputBase,
                              int input0Spatial,
                              int input1Spatial,
                              int outputSpatial,
                              int input0Stride,
                              int input1Stride,
                              int n,
                              int m,
                              int k,
                              float alpha,
                              int st,
                              int end);

    void Run();
};

struct MultiThreadMatMulFloat16SingleOp : MultiThreadBaseOp {
    uint16_t *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;

    MultiThreadMatMulFloat16SingleOp(uint16_t *input0Base,
                                     uint16_t *input1Base,
                                     uint16_t *outputBase,
                                     int input0Spatial,
                                     int input1Spatial,
                                     int outputSpatial,
                                     int input0Stride,
                                     int input1Stride,
                                     int n,
                                     int m,
                                     int k,
                                     float alpha,
                                     int st,
                                     int end);

    void Run();
};

struct MultiThreadMatMulTransBSingleOp : MultiThreadBaseOp {
    float *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;

    MultiThreadMatMulTransBSingleOp(float *input0Base,
                                    float *input1Base,
                                    float *outputBase,
                                    int input0Spatial,
                                    int input1Spatial,
                                    int outputSpatial,
                                    int input0Stride,
                                    int input1Stride,
                                    int n,
                                    int m,
                                    int k,
                                    float alpha,
                                    int st,
                                    int end);
    void Run();
};

struct MultiThreadMatMulTransBFloat16SingleOp : MultiThreadBaseOp {
    uint16_t *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;
    MultiThreadMatMulTransBFloat16SingleOp(uint16_t *input0Base,
                                           uint16_t *input1Base,
                                           uint16_t *outputBase,
                                           int input0Spatial,
                                           int input1Spatial,
                                           int outputSpatial,
                                           int input0Stride,
                                           int input1Stride,
                                           int n,
                                           int m,
                                           int k,
                                           float alpha,
                                           int st,
                                           int end);
    void Run();
};

class CpuMatMulOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMatMulTransBOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuNormalizeOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuSoftMaxOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuSiluOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuTanHOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuReluOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuSigmoidOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

float erf(float a);

class CpuGeluOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuGeluNewOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuSwigluOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

struct MultiThreadSwigluOp : MultiThreadBaseOp {
    float *input, *output;
    int mid, len, n, inputStride, outputStride;

    MultiThreadSwigluOp(float *input, int mid, int len, float *output, int n, int inputStride, int outputStride);

    void Run();
};

struct MultiThreadSwigluFloat16Op : MultiThreadBaseOp {
    uint16_t *input, *output;
    int mid, len, n, inputStride, outputStride;

    MultiThreadSwigluFloat16Op(uint16_t *input, int mid, int len, uint16_t *output, int n, int inputStride, int outputStride);

    void Run();
};

void DoCpuSwigluReshape(Data &input, Data &output);
void DoCpuSwiglu(Data &input, Data &output);

class CpuMulOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAddOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMulToOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

struct MultiThreadAddToFloatOp : MultiThreadBaseOp {
    float *input, *output;
    int len;
    float alpha;

    MultiThreadAddToFloatOp(float *input, float *output, int len, float alpha);

    void Run();
};

static void RunMultiThreadAddToFloat(float *output, float *input, float alpha, int len, AliveThreadPool *pool);

class CpuAddToOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAttentionMaskOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAttentionExtendedMaskOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAlibiMaskOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuTopKOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};