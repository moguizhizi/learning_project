#include "fastllm.h"
#include "qwen3.h"

basellm *CreateModelWithType(const std::string &model_type) {
    basellm *model = nullptr;
    if (model_type == "Qwen3") {
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