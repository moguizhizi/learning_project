#pragma once // 或者用 #ifndef 方式

#include "alivethreadpool.h"
#include "device.h"
#include "struct_space.hpp"

class CpuDevice : BaseDevice {
   public:
    CpuDevice();
};

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

class CpuSplitOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
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

class CpuMulOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAddOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMulToOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

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

class CpuPermuteOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMergeMOE : BaseOperator {
   protected:
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuPermuteSelfOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    bool IsSameCase(const std::vector<int> &axis, const std::vector<int> &dims);
};

class CpuSoftmaxBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuCatDirectBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAppendKVCacheBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuSplitBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuCatBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMulBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMatMulBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMatMulTransBBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuAttentionBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

float erf(float a);

void DoCpuLinearReshape(Data &input, Data &weight, Data &output);
void DoCpuLinear(Data &input, Data &weight, const Data &bias, Data &output);
void DoCpuCatDirect(Data &input0, Data &input1, int axis);

// float的input, int8的weight, 直接计算得到float的output
void Int8LinearPart(
    float *inputData, uint8_t *weightData, float *biasData, float *outputData, LowBitConfig *configs, int n, int m, int k, int st, int end);

// float的input, int4g的weight, 直接计算得到float的output
void Int4GroupLinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData, LowBitConfig *configs, int n, int m, int k,
    int st, int end, int group, int groupCnt);

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

    MultiThreadLinearInt4Op(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride, int *weightSums, int *weightZeros,
        float *scales, float *bias, LowBitConfig *config, int *inputSums);

    void Run();
};

// a = [n, m], b = [k, m], c = aT(b') = [n, k]
void MultiplyInt4MultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int *weightSums, int *weightZeros, float *scales,
    float *bias, std::vector<LowBitConfig> &configs, int threadNum);

void GetArrayMinMax(float *a, int len, float &minValue, float &maxValue);

void QuantizationAll(float *fValue, uint8_t *uValue, int len, LowBitConfig *config);

struct MultiThreadOnlineQuantizationOp : MultiThreadBaseOp {
    float *input;
    uint8_t *output;
    LowBitConfig *configs;
    int n, m, group, groupCnt;
    float *inputSums, *iscales, *izeros;
    int permuteType;

    MultiThreadOnlineQuantizationOp(float *input, uint8_t *output, LowBitConfig *configs, int n, int m, int group, int groupCnt,
        float *inputSums, float *iscales, float *izeros, int permuteType);

    void Run();
};

void OnlineQuantization(float *inputData, std::vector<uint8_t> &uinput, std::vector<LowBitConfig> &inputConfigs, int n, int m, int group,
    int groupCnt, std::vector<float> &inputSums, std::vector<float> &iscales, std::vector<float> &izeros, int permuteType);

struct MultiThreadLinearInt4NoZeroOp : MultiThreadBaseOp {
    uint8_t *a, *b;
    int32_t *c;
    int n, m, k, kstride;
    int *weightSums;
    float *weightMins, *scales, *bias;
    LowBitConfig *config;
    float *inputSums;

    MultiThreadLinearInt4NoZeroOp(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride, int *weightSums, float *weightMins,
        float *scales, float *bias, LowBitConfig *config, float *inputSums);

    void Run();
};

void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, int *weightSums, float *weightMins, float *scales,
    float *bias, std::vector<float> &inputSums, std::vector<float> &iscales, std::vector<float> &izeros, std::vector<LowBitConfig> &configs,
    int startTid, int threadNum, int group, int groupCnt, std::vector<MultiThreadBaseOp *> &ops, AliveThreadPool *pool);

struct MultiThreadSliceOp : MultiThreadBaseOp {
    uint8_t *input, *output;
    int outer, inputStride, outputStride, copyLen;

    MultiThreadSliceOp(uint8_t *output, uint8_t *input, int outer, int outputStride, int inputStride, int copyLen);

    void Run();
};

static void RunMultiThreadSlice(
    uint8_t *output, uint8_t *input, int outer, int inputStride, int outputStride, int copyLen, AliveThreadPool *pool);

struct MultiThreadMatMulSingleOp : MultiThreadBaseOp {
    float *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;

    MultiThreadMatMulSingleOp(float *input0Base, float *input1Base, float *outputBase, int input0Spatial, int input1Spatial, int outputSpatial,
        int input0Stride, int input1Stride, int n, int m, int k, float alpha, int st, int end);

    void Run();
};

struct MultiThreadMatMulFloat16SingleOp : MultiThreadBaseOp {
    uint16_t *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;

    MultiThreadMatMulFloat16SingleOp(uint16_t *input0Base, uint16_t *input1Base, uint16_t *outputBase, int input0Spatial, int input1Spatial,
        int outputSpatial, int input0Stride, int input1Stride, int n, int m, int k, float alpha, int st, int end);

    void Run();
};

struct MultiThreadMatMulTransBSingleOp : MultiThreadBaseOp {
    float *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;

    MultiThreadMatMulTransBSingleOp(float *input0Base, float *input1Base, float *outputBase, int input0Spatial, int input1Spatial,
        int outputSpatial, int input0Stride, int input1Stride, int n, int m, int k, float alpha, int st, int end);
    void Run();
};

struct MultiThreadMatMulTransBFloat16SingleOp : MultiThreadBaseOp {
    uint16_t *input0Base, *input1Base, *outputBase;
    int input0Spatial, input1Spatial, outputSpatial;
    int input0Stride, input1Stride, n, m, k;
    float alpha;
    int st, end;
    MultiThreadMatMulTransBFloat16SingleOp(uint16_t *input0Base, uint16_t *input1Base, uint16_t *outputBase, int input0Spatial,
        int input1Spatial, int outputSpatial, int input0Stride, int input1Stride, int n, int m, int k, float alpha, int st, int end);
    void Run();
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

struct MultiThreadAddToFloatOp : MultiThreadBaseOp {
    float *input, *output;
    int len;
    float alpha;

    MultiThreadAddToFloatOp(float *input, float *output, int len, float alpha);

    void Run();
};

static void RunMultiThreadAddToFloat(float *output, float *input, float alpha, int len, AliveThreadPool *pool);

bool TrySwapLastTwoDimsAndTranspose(Data &input, const std::vector<int> &newDims, const std::vector<int> &inputDims);

bool TransposeSpecialCase(const std::vector<int> &axis, Data &input, const std::vector<int> &inputDims, const std::vector<int> &newDims);

void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m);

void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m);

struct MultiThreadTransposeOp : MultiThreadBaseOp {
    float *pDst, *pSrc;
    int dstStride, srcStride, n, m;

    MultiThreadTransposeOp(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m);

    void Run();
};

struct MultiThreadSiluOp : MultiThreadBaseOp {
    float *input, *output;
    int mid, len, n, inputStride, outputStride;

    MultiThreadSiluOp(float *input, int len, float *output, int n, int inputStride, int outputStride);

    void Run();
};

void SiluMultiThread(float *input, int len, float *output, int n, int inputStride, int outputStride, AliveThreadPool *pool);

float gelu(float x);

struct MultiThreadGeluOp : MultiThreadBaseOp {
    float *input, *output;
    int len, n, inputStride, outputStride;

    MultiThreadGeluOp(float *input, int len, float *output, int n, int inputStride, int outputStride);
    void Run();
};

void GeluMultiThread(float *input, int len, float *output, int n, int inputStride, int outputStride, AliveThreadPool *pool);
void SwigluMultiThread(float *input, int mid, int len, float *output, int n, int inputStride, int outputStride, AliveThreadPool *pool);
void SwigluMultiThreadFloat16(
    uint16_t *input, int mid, int len, uint16_t *output, int n, int inputStride, int outputStride, AliveThreadPool *pool);

float *MOEConvertToFloat32(const Data &src, std::vector<float> &buffer);
std::vector<std::pair<float, int>> CpuComputeRouterScores(const float *logits, const float *bias, int m);
std::vector<int> CpuSelectTopExperts(std::vector<std::pair<float, int>> &routerScores, int topk);
std::vector<ExpertRoute> CpuNormalizeExpertWeights(
    const float *logits, const std::vector<int> &selectedExperts, float routeScale, bool needNorm);
std::vector<ExpertRoute> CpuRouteMoE(
    const float *logits, const float *bias, int m, int topk, float routeScale, bool needNorm, int sharedExpertIndex, float *sharedScale);

void BuildExpertTasks(std::unordered_map<int, std::pair<ExpertRoute, std::vector<int>>> &expertTasks, int bs, const float *fp32logits,
    const float *fp32bias, int num_expert, int topk, float routeScale, bool needNorm, int SharedExpertIndex, float *sharedScale);

void PrepareTempInput(Data &tempInput, const Data &input, const std::vector<int> &indices, int m, int uintsize, AliveThreadPool *pool);
void ExpertForwardUp(Data &w3, Data &tempInput, Data &upWeight, Data &upBias);
void ExpertApplySwiglu(Data &w1, Data &w3, AliveThreadPool *pool);
float *ExpertForwardDown(Data &w1, Data &downWeight, Data &downBias, Data &w2);
float *RunSingleExpertForward(const std::pair<ExpertRoute, std::vector<int>> &expertTask, const Data &input, Data **weights, Data &w1, Data &w2,
    Data &w3, AliveThreadPool *pool);
void RunMoeReduceAndAccumulate(const std::pair<ExpertRoute, std::vector<int>> &expertTask, std::vector<float> *tempResult, float *curOutput,
    int dim, AliveThreadPool *pool);
