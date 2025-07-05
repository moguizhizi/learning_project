#pragma once

#include "struct_space.hpp"
#include <string>

class basellm {
  public:
    basellm();
    ~basellm();

    std::string model_type;

    int block_cnt;
    int rotary_dim;
    int head_dim = 0;

    int bos_token_id;
    int eos_token_id;

    int embed_dim;
    int num_attention_heads = 0;

    int num_key_value_heads;

    WeightMap weight;

    std::vector<WeightMergeRule> weightMergeRules;

    void InitParams();
};