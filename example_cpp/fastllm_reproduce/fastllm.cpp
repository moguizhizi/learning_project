#include "fastllm.h"
#include "file_utils.hpp"
#include "qwen3.h"
#include <algorithm>
#include <cstring>
#include <regex>

basellm *CreateModelWithType(const std::string &model_type) {
    basellm *model = nullptr;
    if (model_type == "qwen3") {
        model = new Qwen3Model();
    }

    return model;
}

void AddDictRecursion(basellm *model, const std::string &prefix, const json11::Json &config) {
    for (auto &it : config.object_items()) {
        if (it.second.is_object()) {
            AddDictRecursion(model, prefix + it.first + ".", it.second);
        } else {
            model->weight.AddDict(prefix + it.first, it.second.is_string() ? it.second.string_value() : it.second.dump());
        }
    }
}

bool StringEndWith(const std::string &s, const std::string &end) { return s.size() >= end.size() && s.substr(s.size() - end.size()) == end; }
bool StringStartWith(const std::string &s, const std::string &end) { return s.size() >= end.size() && s.substr(0, end.size()) == end; }
void ParseDataType(std::string weightName, const std::vector<std::pair<std::string, std::string>> &dtypeRules, DataType &dataType, int &groupCnt) {
    std::string matchedType = "";
    for (int i = 0; i < dtypeRules.size(); i++) {
        std::regex pattern(dtypeRules[i].first);
        if (std::regex_search(weightName, pattern)) {
            matchedType = dtypeRules[i].second;
        }
    }
    transform(matchedType.begin(), matchedType.end(), matchedType.begin(), ::tolower);
    if (matchedType != "") {
        for (auto &it : dataTypeNames) {
            for (auto &dataTypeName : it.second) {
                if (DefaultGroupCnts.find(it.first) != DefaultGroupCnts.end()) {
                    if (StringStartWith(matchedType, dataTypeName)) {
                        dataType = it.first;
                        if (matchedType != dataTypeName) {
                            groupCnt = std::atoi(matchedType.substr(dataTypeName.size()).c_str());
                        } else {
                            groupCnt = DefaultGroupCnts[it.first];
                        }
                    }
                } else {
                    if (matchedType == dataTypeName) {
                        dataType = it.first;
                    }
                }
            }
        }
    }
}

uint32_t as_uint(const float x) { return *(uint32_t *)&x; }
float as_float(const uint32_t x) { return *(float *)&x; }

float half_to_float(const uint16_t x) {         // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0,
                                                // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint32_t e = (x & 0x7C00) >> 10;      // exponent
    const uint32_t m = (x & 0x03FF) << 13;      // mantissa
    const uint32_t v = as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                    ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
}
void ConvertDataType(uint8_t *src, DataType srcDtype, uint8_t *dst, DataType dstDtype, uint64_t len) {
    if (srcDtype == dstDtype) {
        int unitSize = 4;
        if (dstDtype == DataType::FLOAT32) {
            unitSize = 4;
        } else if (dstDtype == DataType::BFLOAT16 || dstDtype == DataType::FLOAT16) {
            unitSize = 2;
        } else {
            ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
        }
        std::memcpy(dst, src, unitSize * len);
    } else if (srcDtype == DataType::FP8_E4M3 && dstDtype == DataType::FLOAT16) {
        ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
    } else if (srcDtype == DataType::BFLOAT16 && dstDtype == DataType::FLOAT32) {
        uint16_t *u16src = (uint16_t *)src;
        uint16_t *u16dst = (uint16_t *)dst;

        for (int i = 0; i < len; i++) {
            u16dst[2 * i] = 0;
            u16dst[2 * i + 1] = u16src[i];
        }
    } else if (srcDtype == DataType::FLOAT16 && dstDtype == DataType::FLOAT32) {
        uint16_t *u16src = (uint16_t *)src;
        float *fdst = (float *)dst;

        for (int i = 0; i < len; i++) {
            fdst[i] = half_to_float(u16src[i]);
        }
    } else {
        ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
    }
}

void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
    if (n < 4 || m < 4) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }

        return;
    }
}

void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
    int per = 4;
    for (int i = 0; i < n; i += per) {
        for (int j = 0; j < m; j += per) {
            Transpose4x4(pDst + j * dstStride + i, pSrc + i * srcStride + j, dstStride, srcStride, std::min(per, n - i), std::min(per, m - j));
        }
    }
}