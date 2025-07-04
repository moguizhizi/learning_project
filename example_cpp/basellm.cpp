#include "basellm.h"
#include "qwen3.h"

basellm::basellm(){

}

basellm::~basellm(){
    
}

void basellm::InitParams(){
    if(this->weight.dicts.find("model_type")!=this->weight.dicts.end()){
        this->model_type = this->weight.dicts["model_type"].c_str();
    }

    if(this->weight.dicts.find("bos_token_id")!=this->weight.dicts.end()){
        this->bos_token_id = atoi(this->weight.dicts["bos_token_id"].c_str());
    }

    if(this->weight.dicts.find("eos_token_id")!=this->weight.dicts.end()){
        this->eos_token_id = atoi(this->weight.dicts["eos_token_id"].c_str());
    }

    if(this->weight.dicts.find("hidden_size")!=this->weight.dicts.end()){
        this->embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
    }

    if(this->weight.dicts.find("num_hidden_layers")!=this->weight.dicts.end()){
        this->block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
    }

    if(this->weight.dicts.find("head_dim")!=this->weight.dicts.end()){
        this->head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        this->rotary_dim = this->head_dim;
    }

    if(this->weight.dicts.find("num_attention_heads")!=this->weight.dicts.end()){
        this->num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
    }

}