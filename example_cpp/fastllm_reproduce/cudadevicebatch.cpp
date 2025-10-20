#include "cudadevicebatch.h"
#include "cudadevice.h"
#include "fastllm-cuda.cuh"
#include "file_utils.hpp"

void CudaCatBatchOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data **inputs = (Data **)(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int dimsLen = inputs[0]->dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int part = intParams.find("input___batch")->second;
    std::vector<int> dims = inputs[0]->dims;
    dims[axis] = part;
    output.dataType = inputs[0]->dataType;
    output.Resize(dims);
}

void CudaCatBatchOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data **inputs = (Data **)(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int dimsLen = inputs[0]->dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int part = intParams.find("input___batch")->second;
    output.Allocate();
    FastllmCudaCatBatch(inputs, output, axis);
}

void CudaSplitBatchOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data **outputs = (Data **)(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int part = input.dims[axis];
    for (int i = 0; i < part; i++) {
        outputs[i]->Allocate();
    }
    FastllmCudaSplitBatch(input, outputs, axis);
}

void CudaSplitBatchOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data **outputs = (Data **)(datas.find("output")->second);
    int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int part = input.dims[axis];
    std::vector<int> dims = input.dims;
    dims[axis] = 1;
    for (int i = 0; i < part; i++) {
        outputs[i]->dataType = input.dataType;
        outputs[i]->Resize(dims);
    }
}

void DoCudaAttentionBatch(Data **qs, Data **ks, Data **vs, Data **masks, Data **outputs, int group, float scale, int batch) {
    long long aveLen = 0;
    for (int i = 0; i < batch; i++) {
        aveLen += ks[i]->dims[1];
    }
    aveLen /= batch;
    if (qs[0]->dataType == DataType::FLOAT32 || true) {
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
        }
        FastllmCudaAttentionBatch(qs, ks, vs, masks, outputs, group, scale, batch);
    } else {
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
        }
        for (int i = 0; i < batch; i++) {
            if (qs[i]->dataType == DataType::FLOAT16) {
                if (masks == nullptr || masks[i] == nullptr) {
                    FastllmCudaHalfAttention(*qs[i], *ks[i], *vs[i], Data(), *outputs[i], group, scale, 0);
                } else {
                    FastllmCudaHalfAttention(*qs[i], *ks[i], *vs[i], *masks[i], *outputs[i], group, scale, 0);
                }
            } else {
                ErrorInFastLLM("AttentionBatch: datatype should be float32 or float16.");
            }
        }
    }
}
