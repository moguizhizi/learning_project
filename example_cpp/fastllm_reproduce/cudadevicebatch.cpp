#include "cudadevicebatch.h"

#include "cudadevice.h"
#include "device.h"
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

void CudaMulBatchOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data **inputs = (Data **)(datas.find("input")->second);
    Data **outputs = (Data **)(datas.find("output")->second);

    float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
    int batch = intParams.find("input___batch")->second;
    if (outputs[0]->Count(0) > outputs[0]->expansionSize) {
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
            AssertInFastLLM(inputs[i]->dataType == DataType::FLOAT32, "Mul error: Data's type should be float32.\n");
        }
    }

    FastllmCudaMulBatch(inputs, v, batch, outputs);
}

void CudaMatMulTransBBatchOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    int batch = intParams.find("input0___batch")->second;
    Data **input0s = ((Data **)datas.find("input0")->second);
    Data **input1s = ((Data **)datas.find("input1")->second);
    Data **outputs = ((Data **)datas.find("output")->second);

    if (input0s[0]->dims.size() == 3 && input1s[0]->dims.size() == 3) {
        AssertInFastLLM(input0s[0]->dataType == DataType::FLOAT32 && input1s[0]->dataType == DataType::FLOAT32,
            "MatMul's input's type should be float32.\n");
        AssertInFastLLM(input0s[0]->dims[0] == input1s[0]->dims[0] && input0s[0]->dims[2] == input1s[0]->dims[2], "MatMul's shape error.\n");
        for (int i = 0; i < batch; i++) {
            outputs[i]->dataType = input0s[i]->dataType;
            outputs[i]->Resize({input0s[i]->dims[0], input0s[i]->dims[1], input1s[i]->dims[1]});
        }
    } else {
        // BaseOperator *op = (BaseOperator *)(new CudaMatMulTransBOp());
        // DataDict tempDatas = datas;
        // for (int i = 0; i < batch; i++) {
        //     tempDatas["input0"] = ((Data **)datas.find("input0")->second)[i];
        //     tempDatas["input1"] = ((Data **)datas.find("input1")->second)[i];
        //     tempDatas["output"] = ((Data **)datas.find("output")->second)[i];
        //     op->Reshape("MatMulTransB", tempDatas, floatParams, intParams);
        // }
        // delete op;
    }
}

void CudaMatMulTransBBatchOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    int batch = intParams.find("input0___batch")->second;
    Data **input0s = (Data **)(datas.find("input0")->second);
    Data **input1s = (Data **)(datas.find("input1")->second);
    Data **outputs = (Data **)(datas.find("output")->second);
    float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;

    std::vector<void *> i0s, i1s, os;
    std::vector<int> ns, ms, ks, i0Strides, i1Strides;
    for (int i = 0; i < batch; i++) {
        auto &input0 = *input0s[i];
        auto &input1 = *input1s[i];
        auto &output = *outputs[i];
        output.Allocate();
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        for (int b = 0; b < batch0; b++) {
            i0s.push_back((float *)input0.cudaData + b * input0Spatial);
            i1s.push_back((float *)input1.cudaData + b * input1Spatial);
            os.push_back((float *)output.cudaData + b * outputSpatial);
            ns.push_back(n);
            ms.push_back(m);
            ks.push_back(k);
            i0Strides.push_back(input0Stride);
            i1Strides.push_back(input1Stride);
        }
    }

    FastllmCudaBatchMatMulTransBBatch(
        i0s.data(), i1s.data(), os.data(), ns.data(), ms.data(), ks.data(), i0Strides.data(), i1Strides.data(), alpha, (int)i0s.size());
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

void DoCudaCatDirectBatch(Data **input0s, Data **input1s, int batch, int axis) {
    std::vector<void *> dsts, srcs;
    std::vector<size_t> dpitchs, spitchs, widths, heights;
    dsts.resize(batch);
    srcs.resize(batch);
    dpitchs.resize(batch);
    spitchs.resize(batch);
    widths.resize(batch);
    heights.resize(batch);

    for (int b = 0; b < batch; b++) {
        Data &input0 = *input0s[b];
        Data &input1 = *input1s[b];

        if (input0.dims.size() == 0) {
            input0.Resize(input1.dims);
            AssertInFastLLM(input0.expansionDims.size() == input1.dims.size() && input1.dims[axis] <= input0.expansionDims[axis],
                "CatDirect Error: input0's expansion size is not enough.\n");
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis);
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;

            dsts[b] = (input0.cudaData);
            dpitchs[b] = (input0Stride * unitSize);
            srcs[b] = (input1.cudaData);
            spitchs[b] = (input1Stride * unitSize);
            widths[b] = (input1.dims[axis] * inner * unitSize);
            heights[b] = (outer);
            continue;
        }

        int oldAxisDims = input0.dims[axis];
        input0.dims[axis] += input1.dims[axis];
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        dsts[b] = ((uint8_t *)input0.cudaData + oldAxisDims * inner * unitSize);
        dpitchs[b] = (input0Stride * unitSize);
        srcs[b] = (input1.cudaData);
        spitchs[b] = (input1Stride * unitSize);
        widths[b] = (input1.dims[axis] * inner * unitSize);
        heights[b] = (outer);
    }

    FastllmCudaMemcpy2DDeviceToDeviceBatch(dsts.data(), dpitchs.data(), srcs.data(), spitchs.data(), widths.data(), heights.data(), dsts.size());
}

void DoCudaAttentionBatchReshape(Data **qs, Data **vs, Data **outputs, int batch) {
    for (int i = 0; i < batch; i++) {
        outputs[i]->dataType = qs[i]->dataType;
        outputs[i]->Resize({qs[i]->dims[0], qs[i]->dims[1], vs[i]->dims[2]});
    }
}