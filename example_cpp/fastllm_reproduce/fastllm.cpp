#include "fastllm.h"
#include "qwen3.h"
#include <algorithm>
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