#pragma once // 或者用 #ifndef 方式

#include "basellm.h"
#include "cpudevice.h"
#include "device.h"
#include "struct_space.hpp"

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
void DoCudaLinear(Data &input, Data &weight, const Data &bias, Data &output);
void DoCudaSplitReshape(Data &input, int axis, int start, int end, Data &output);
void DoCudaSplit(Data &input, int axis, int start, int end, Data &output);
void DoCudaPermuteSelf(Data &input, const std::vector<int> &axis);
void DoCudaCatDirect(Data &input0, Data &input1, int axis);
void DoCudaAttentionReshape(Data &q, Data &v, Data &output);
void DoCudaAttention(Data &q, Data &k, Data &v, Data &mask, Data &output, int group, float scale, int maskType);

void DoCudaSwigluReshape(Data &input, Data &output);
void DoCudaSwiglu(Data &input, Data &output);

std::vector<std::pair<float, int>> ComputeRouterScores(const float *logits, const float *bias, int m);
std::vector<int> SelectTopExperts(std::vector<std::pair<float, int>> &routerScores, int topk);
std::vector<ExpertRoute> NormalizeExpertWeights(const float *logits, const std::vector<int> &selectedExperts, float routeScale, bool needNorm);
std::vector<ExpertRoute> RouteMoE(const float *logits, const float *bias, int m, int topk, float routeScale, bool needNorm, float *sharedScale);
void DoCudaMergeMOE(Data &input, Data &output, Data &gateBias, Data &logits, Data &w1, Data &w2, Data &w3, Data **weights, Data **biass,
    int topk, int needNorm, float sharedScale, float routeScale);

void DoCudaCatDirectBatch(Data **input0s, Data **input1s, int batch, int axis);
void DoCudaAttentionBatchReshape(Data **qs, Data **vs, Data **outputs, int batch);
void DoCudaAttentionBatch(Data **qs, Data **ks, Data **vs, Data **masks, Data **outputs, int group, float scale, int batch);

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

class CudaMatMulTransBOp : public BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaMatMulTransBBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaConv2DOp : CpuConv2DOp {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaRepeatOp : CpuRepeatOp {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaReluOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaAlibiMaskOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CpuMergeMOE : BaseOperator {
   protected:
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaMergeAttention : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaLayerNormOp : BaseOperator {
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class CudaRMSNormOp : BaseOperator {
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};