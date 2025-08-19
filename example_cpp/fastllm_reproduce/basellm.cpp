#include "basellm.h"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

#include "fastllm.h"
#include "file_utils.hpp"
#include "qwen3.h"
#include "utils.h"
#include <cstring>

basellm::basellm() {}

basellm::~basellm() {}

void basellm::InitParams() {
    if (this->weight.dicts.find("model_type") != this->weight.dicts.end()) {
        this->model_type = this->weight.dicts["model_type"].c_str();
    }

    if (this->weight.dicts.find("bos_token_id") != this->weight.dicts.end()) {
        this->bos_token_id = atoi(this->weight.dicts["bos_token_id"].c_str());
    }

    if (this->weight.dicts.find("eos_token_id") != this->weight.dicts.end()) {
        this->eos_token_id = atoi(this->weight.dicts["eos_token_id"].c_str());
    }

    if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
        this->embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
    }

    if (this->weight.dicts.find("num_hidden_layers") != this->weight.dicts.end()) {
        this->block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
    }

    if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
        this->head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        this->rotary_dim = this->head_dim;
    }

    if (this->weight.dicts.find("num_attention_heads") != this->weight.dicts.end()) {
        this->num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
    }

    this->embed_dim = this->head_dim * this->num_attention_heads;

    this->num_key_value_heads = this->num_attention_heads;
    if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
        this->num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
    }
}

std::map<std::string, std::vector<std::pair<std::string, DataType>>> basellm::GetTensorMap(const std::vector<std::string> &tensorNames) {
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> ret;
    for (auto &name : tensorNames) {
        std::string realName = name;
        if (StringEndWith(name, "qweight")) {
            realName = name.substr(0, name.size() - 7) + "weight";
        }

        WeightType weightType = this->weight.GetWeightType(realName);

        DataType dataType = DataType::DATA_AUTO_NONE;
        if (weightType == WeightType::LINEAR) {
            dataType = DataType::DATA_AUTO_LINEAR;
            if (this->cantQuantLinears.find(realName) != this->cantQuantLinears.end()) {
                dataType = DataType::BFLOAT16;
            }
        } else if (weightType == WeightType::EMBEDDING) {
            dataType = DataType::DATA_AUTO_EMBEDDING;
        }
        ret[name].push_back(std::make_pair(realName, dataType));
    }

    return ret;
}

std::map<std::string, std::vector<std::pair<std::string, DataType>>>
basellm::GetTensorMap(const std::vector<std::string> &tensorNames, bool useMoeDataType, DataType moeDataType) {
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> ret;

    for (auto &name : tensorNames) {
        std::string realName = name;
        if (StringEndWith(name, ".qweight")) {
            realName = name.substr(0, name.size() - 7) + "weight";
        }

        WeightType weightType = this->weight.GetWeightType(realName);
        DataType dataType = DataType::DATA_AUTO_NONE;

        if (weightType == WeightType::LINEAR) {
            dataType = DataType::DATA_AUTO_LINEAR;
            if (this->cantQuantLinears.find(realName) != this->cantQuantLinears.end()) {
                dataType = DataType::FLOAT16;
            }
        } else if (weightType == WeightType::EMBEDDING) {
            dataType = DataType::DATA_AUTO_EMBEDDING;
        }

        // 如果是 MoE 并且开启 useMoeDataType，则替换类型
        if (useMoeDataType && this->moeLinears.find(realName) != this->moeLinears.end()) {
            dataType = moeDataType;
        }

        ret[name].push_back(std::make_pair(realName, dataType));
    }

    return ret;
}

void basellm::MergeWeightsFromRules(const std::string &weightName,
                                    const std::set<std::string> &allWeightNames,
                                    const std::set<std::string> &allFinishName,
                                    bool &needMerge) {
    for (auto &rule : this->weightMergeRules) {
        if (rule.allInputs.find(weightName) == rule.allInputs.end()) {
            continue;
        }

        needMerge = true;

        bool canMerge = true;
        for (auto &input : rule.allInputs) {
            if (allWeightNames.find(input) == allWeightNames.end() || allFinishName.find(input) == allFinishName.end()) {
                canMerge = false;
                break;
            }
        }

        if (!canMerge) {
            continue;
        }

        for (auto &it : rule.rules) {
            DataType dataType = this->weight[it.inputs[0]].dataType;
            int dimSize = this->weight[it.inputs[0]].dims.size();
            int dim0size = this->weight[it.inputs[0]].dims[0];
            int groupCnt = this->weight[it.inputs[0]].groupCnt;
            int blockK = this->weight[it.inputs[0]].blockK;
            int blockM = this->weight[it.inputs[0]].blockM;

            for (auto &input : it.inputs) {
                if (dataType != this->weight[input].dataType || dimSize != this->weight[input].dims.size() ||
                    dim0size != this->weight[input].dims[0]) {
                    canMerge = false;
                    break;
                }

                if (dimSize == 2) {
                    if (groupCnt != -1 && this->weight[input].dims[1] % groupCnt != 0) {
                        canMerge = false;
                        break;
                    }

                    if (blockK != -1 && this->weight[input].dims[0] % blockK != 0) {
                        canMerge = false;
                        break;
                    }

                    if (blockM != -1 && this->weight[input].dims[1] % blockM != 0) {
                        canMerge = false;
                        break;
                    }
                }
            }
        }

        if (!canMerge) {
            continue;
        }

        for (auto &it : rule.rules) {
            if (allWeightNames.find(it.inputs[0]) == allWeightNames.end()) {
                continue;
            }

            DataType dataType = this->weight[it.inputs[0]].dataType;
            int dim0Len = 0;
            for (auto &input : it.inputs) {
                dim0Len += this->weight[input].dims[0];
            }

            std::string mergeName = it.output;

            if (this->weight[it.inputs[0]].dims.size() == 1) {
                // 一维权重合并
                this->weight[mergeName] = Data(dataType, {dim0Len});
                Data &mergeData = this->weight[mergeName];
                mergeData.name = mergeName;
                mergeData.Allocate();
                int offset = 0;
                for (auto &input : it.inputs) {
                    std::memcpy(mergeData.cpuData + offset, this->weight[input].cpuData, this->weight[input].GetBytes());
                    offset += this->weight[input].GetBytes();
                }
            } else {
                // 二维权重合并
                this->weight[mergeName] = Data(dataType, {dim0Len, this->weight[it.inputs[0]].dims[1]});
                Data &mergeData = this->weight[mergeName];
                mergeData.name = mergeName;
                mergeData.group = this->weight[it.inputs[0]].group;
                mergeData.groupCnt = this->weight[it.inputs[0]].groupCnt;
                mergeData.perChannelAxis = this->weight[it.inputs[0]].perChannelAxis;
                mergeData.blockK = this->weight[it.inputs[0]].blockK;
                mergeData.blockM = this->weight[it.inputs[0]].blockM;

                mergeData.Allocate();
                int offset = 0;
                for (auto &input : it.inputs) {
                    mergeData.perChannelsConfigs = AppendVector(mergeData.perChannelsConfigs, this->weight[input].perChannelsConfigs);
                    mergeData.scales = AppendVector(mergeData.scales, this->weight[input].scales);
                    mergeData.mins = AppendVector(mergeData.mins, this->weight[input].mins);
                    mergeData.zeros = AppendVector(mergeData.zeros, this->weight[input].zeros);
                    mergeData.halfScales = AppendVector(mergeData.halfScales, this->weight[input].halfScales);

                    std::memcpy(mergeData.cpuData + offset, this->weight[input].cpuData, this->weight[input].GetBytes());
                    offset += this->weight[input].GetBytes();
                }

                mergeData.CalcWeightSum();
#if defined(USE_TFACC) || defined(USE_NUMA)
                try {
                    std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                    if (s != "" && s != "OFF") {
                        if (model->specialWeights.find(mergeName) != model->specialWeights.end()) {
                            mergeData.weightSum.resize(1);
                            RegisterFastllmData(&mergeData, it.type);
                        }
                    }
                } catch (...) {
                }
#endif
            }

            for (auto &input : it.inputs) {
                this->weight.weight.erase(input);
            }
        }
    }
}
