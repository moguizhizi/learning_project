#include "fastllm.h"
#include "qwen3.h"

basellm *CreateModelWithType(const std::string &model_type) {
    basellm *model = nullptr;
    if (model_type == "Qwen3") {
        model = new Qwen3Model();
    }

    return model;
}