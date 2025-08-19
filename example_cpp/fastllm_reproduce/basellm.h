#pragma once

#include "common_class.h"
#include "enum_space.h"
#include "struct_space.hpp"
#include "types.h"
#include <string>

class basellm {
  public:
    basellm();
    ~basellm();

    virtual void WarmUp() {}; // 预热

    std::string model_type;
    std::set<std::string> cantQuantLinears;
    std::set<std::string> moeLinears;
    std::set<int> eos_token_ids;

    int block_cnt;
    int rotary_dim;
    int head_dim = 0;

    std::string pre_prompt;                       // 最初对话的提示语
    std::string user_role, bot_role, history_sep; // 用于生成每一轮的prompt

    int bos_token_id;
    int eos_token_id;

    int embed_dim;
    int num_attention_heads = 0;

    int num_key_value_heads;

    WeightMap weight;

    std::vector<WeightMergeRule> weightMergeRules;
    std::map<std::string, std::string> specialWeights; // 一些特殊层，可以提前注册（一般用于TFACC）

    Data sinData;
    Data cosData;

    void InitParams();
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> GetTensorMap(const std::vector<std::string> &tensorNames);
    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    GetTensorMap(const std::vector<std::string> &tensorNames, bool useMoeDataType, DataType moeDataType);
    void MergeWeightsFromRules(const std::string &weightName,
                               const std::set<std::string> &allWeightNames,
                               const std::set<std::string> &allFinishName,
                               bool &needMerge);
};

class BaseDevice {};
class CpuDevice : BaseDevice {};