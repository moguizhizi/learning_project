#pragma once

#include "basellm.h"
#include <string>

static std::map<DataType, std::vector<std::string>> dataTypeNames = {{DataType::FLOAT32, {"float32", "fp32"}},
                                                                     {DataType::BFLOAT16, {"bfloat32", "bf32"}},
                                                                     {DataType::INT16, {"int16"}},
                                                                     {DataType::INT8, {"int8"}},
                                                                     {DataType::INT4, {"int4o"}},
                                                                     {DataType::INT2, {"int2"}},
                                                                     {DataType::BIT, {"bit"}},
                                                                     {DataType::FLOAT16, {"float16", "fp16", "half"}},
                                                                     {DataType::INT4_NOZERO, {"int4"}},
                                                                     {DataType::INT4_GROUP, {"int4g"}},
                                                                     {DataType::FP8_E4M3, {"float8", "fp8", "fp8_e4m3"}},
                                                                     {DataType::INT2_GROUP, {"int2g"}},
                                                                     {DataType::BASE3_GROUP, {"base3g"}}};

static std::map<DataType, int> DefaultGroupCnts = {{DataType::INT4_GROUP, 128}, {DataType::INT2_GROUP, 128}, {DataType::BASE3_GROUP, 128}};

basellm *CreateModelWithType(const std::string &model_type);
void AddDictRecursion(basellm *model, const std::string &prefix, const json11::Json &config);
bool StringEndWith(const std::string &s, const std::string &end);
bool StringStartWith(const std::string &s, const std::string &end);
void ParseDataType(std::string weightName, const std::vector<std::pair<std::string, std::string>> &dtypeConfig, DataType &datatype, int &groupCnt);