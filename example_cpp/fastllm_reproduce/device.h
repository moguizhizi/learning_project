#include "types.h"
#include <string>

class BaseOperator {
  public:
    // 是否可以运行某一个算子
    virtual bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

    // 对某一个算子进行形状推理
    virtual void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

    // 对某一个算子进行推理
    virtual void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) = 0;
};