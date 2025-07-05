#include "qwen3.h"
#include "struct_space.hpp"
#include <cmath>

Qwen3Model::Qwen3Model() {
    this->model_type = "qwen3";
    this->model_struct = "llama";

    this->block_cnt = 32;
    this->rotary_dim = 128;

    // 默认使用 llama3 的提示词和instruction
    this->pre_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>";
    this->user_role = "<|start_header_id|>user<|end_header_id|>\n";
    this->bot_role = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
    this->history_sep = "<|eot_id|>\n";

    this->weight.embeddingsNames.insert("model.embed_tokens.weight");
    this->weight.linearNames = {"lm_head.weight",
                                "model.layers.*.mlp.down_proj.weight",
                                "model.layers.*.mlp.up_proj.weight",
                                "model.layers.*.mlp.gate_proj.weight",
                                "model.layers.*.mlp.gate_proj.weight",
                                "model.layers.*.mlp.gateup_proj.weight",
                                "model.layers.*.self_attn.o_proj.weight",
                                "model.layers.*.self_attn.q_proj.weight",
                                "model.layers.*.self_attn.k_proj.weight",
                                "model.layers.*.self_attn.v_proj.weight",
                                "model.layers.*.self_attn.mergeqkv.weight",
                                "model.layers.*.self_attn.W_pack.weight"};
}

void Qwen3Model::InitParams() {
    basellm::InitParams();

    if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
        this->max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
    }

    if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
        this->rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
    }

    if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
        this->rope_base = atof(this->weight.dicts["rope_theta"].c_str());
    }

    if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
        this->rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
    }

    for (int i = 0; i < this->block_cnt; i++) {
        std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
        std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
        std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
        this->weightMergeRules.push_back(
            {WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})});

        std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
        std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
        std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
        std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
        std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
        std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
        std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
        std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
        this->weightMergeRules.push_back(
            {WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                              WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})});
    }
}

std::pair<std::vector<float>, std::vector<float>> Qwen3Model::UpdateRotaryPosEmb(float rope_base, float rope_factor, int seqlen) {
    int positions = std::min(this->max_positions, seqlen);
    this->sin.resize(positions);
    this->cos.resize(positions);

    std::vector<float> invFreq;
    for (int i = 0; i < this->rotary_dim; i = i + 2) {
        invFreq.push_back(1.0 / pow(rope_base, i * 1.0 / this->rotary_dim));
    }

    float scale = this->rope_type == RoPEType::LINEAR_SCALE ? rope_factor : 1.0;
    for (int i = 0; i < positions; i++) {
        for (int j = 0; j < invFreq.size(); j++) {
            this->sin[i][j] = ::sin((float)i / scale * invFreq[j]);
            this->cos[i][j] = ::cos((float)i / scale * invFreq[j]);
        }
    }

    std::vector<float> fsin, fcos;
    for (int i = 0; i < positions; i++) {
        fsin.insert(fsin.end(), this->sin[i].begin(), this->sin[i].end());
        fcos.insert(fcos.end(), this->cos[i].begin(), this->cos[i].end());
    }

    return std::make_pair(fsin, fcos);
}