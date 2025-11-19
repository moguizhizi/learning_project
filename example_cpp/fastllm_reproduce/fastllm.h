#pragma once

#include <string>

#include "basellm.h"

#ifndef __CUDACC__
#    if defined(__GNUC__) && __GNUC__ < 8 && !defined(__clang__)
#        include <experimental/filesystem>
#    else
#        include <filesystem>
#    endif
#endif

#ifndef __CUDACC__
#    if (defined(_MSC_VER) && _MSC_VER <= 1900) || (defined(__GNUC__) && __GNUC__ < 8 && !defined(__clang__)) // VS 2015)
namespace fs = std::experimental::filesystem;
#    else
namespace fs = std::filesystem;
#    endif
#endif

#ifndef __CUDACC__
static bool FileExists(std::string filePath) {
#    if defined(__GNUC__) && __GNUC__ < 9
    return access(filePath.c_str(), R_OK) == 0;
#    else
    fs::path path(filePath);
    return fs::exists(path);
#    endif
}
#endif

#include "alivethreadpool.h"

static std::map<DataType, std::vector<std::string>> dataTypeNames = {{DataType::FLOAT32, {"float32", "fp32"}},
    {DataType::BFLOAT16, {"bfloat32", "bf32"}}, {DataType::INT16, {"int16"}}, {DataType::INT8, {"int8"}}, {DataType::INT4, {"int4o"}},
    {DataType::INT2, {"int2"}}, {DataType::BIT, {"bit"}}, {DataType::FLOAT16, {"float16", "fp16", "half"}}, {DataType::INT4_NOZERO, {"int4"}},
    {DataType::INT4_GROUP, {"int4g"}}, {DataType::FP8_E4M3, {"float8", "fp8", "fp8_e4m3"}}, {DataType::INT2_GROUP, {"int2g"}},
    {DataType::BASE3_GROUP, {"base3g"}}};

static std::map<DataType, int> DefaultGroupCnts = {{DataType::INT4_GROUP, 128}, {DataType::INT2_GROUP, 128}, {DataType::BASE3_GROUP, 128}};

template <typename T>
void TransposeSimple(T *pDst, T *pSrc, int dstStride, int srcStride, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            pDst[j * dstStride + i] = pSrc[i * srcStride + j];
        }
    }
}

basellm *CreateModelWithType(const std::string &model_type);
void AddDictRecursion(basellm *model, const std::string &prefix, const json11::Json &config);
bool StringEndWith(const std::string &s, const std::string &end);
bool StringStartWith(const std::string &s, const std::string &end);
void ParseDataType(
    std::string weightName, const std::vector<std::pair<std::string, std::string>> &dtypeConfig, DataType &datatype, int &groupCnt);
void ConvertDataType(uint8_t *src, DataType srcDtype, uint8_t *dst, DataType dstDtype, uint64_t len);
void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m);
void TransposeF32(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m);
std::string GetModelType(const json11::Json &config, bool weightOnly, bool isJsonModel);
void CheckAWQModel(const json11::Json &config, bool &isAwqModel, int &awqGroupCnt);
void SetEosTokenIds(basellm *model, const json11::Json &config, const json11::Json &generation_config);
void SplitString(const std::string &str, const std::set<char> &chars, std::vector<std::string> &ret);
std::string Base64Decode(const std::string &encoded);
void LoadLLMTokenizerFromHFToModel(const std::string &path, basellm *model);
void LoadLoRA(const std::string &loraPath, std::map<std::string, std::pair<std::string, std::string>> &loraDicts, SafeTensors *&loraTensors,
    float &loraScaling);
SafeTensors LoadSafeTensors(const std::string &path);
std::vector<std::pair<std::string, std::string>> ParseDtypeRulesFromConfigFile(const std::string &configPath);
DataType ResolveAutoDataType(const std::string &weightName, const std::vector<std::pair<std::string, std::string>> &dtypeRules,
    DataType dataType, int curGroupCnt, DataType linearDataType, DataType oriDataType, const SafeTensorItem &tensor);
void ApplyLoRAWeight(const std::string &weightName, const std::map<std::string, std::pair<std::string, std::string>> &loraDicts,
    SafeTensors *loraTensors, SafeTensorItem &tensor, DataType oriDataType, float loraScaling);

template <typename T>
static std::vector<T> AppendVector(const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<T> ret = a;
    for (int i = 0; i < b.size(); i++) {
        ret.push_back(b[i]);
    }
    return ret;
}

void barrier();
void ToDataType(const Data &input, DataType dataType);

bool GetCudaEmbedding();
bool GetLowMemMode();

AliveThreadPool *GetAlivePool();

int GetThreads();

void CatBatch(std::vector<Data *> &input, int axis, Data &outputs);
void Split(const Data &input, int axis, int start, int end, Data &output);
void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha = 1.0, int group = 1);
void AddTo(Data &input0, const Data &input1, float alpha = 1.0); // input0 += input1 * alpha
void Mul(const Data &input, float v, Data &output);
void AttentionMask(Data &input, const Data &mask, float maskValue); // 把input里对应位置mask中为1的部分变成maskValue