#include "device.h"
#include "basellm.h"

bool BaseOperator::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) { return true; }

void BaseOperator::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {

    if (datas.find("input") == datas.end() || datas.find("output") == datas.end()) {
        return;
    }

    Data *inputs = datas.find("input")->second;
    Data *outputs = datas.find("output")->second;
    if (inputs == outputs) {
        return;
    }

    outputs[0].dataType = inputs[0].dataType;
    outputs[0].Resize(inputs[0].dims);
}