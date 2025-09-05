#pragma once // 或者用 #ifndef 方式

#include "device.h"

void DoCudaLinearReshape(Data &input, Data &weight, Data &output);

class CudaLinearOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};