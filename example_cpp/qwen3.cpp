#include "qwen3.h"

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
}