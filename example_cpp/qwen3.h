#pragma once

#include "basellm.h"

class Qwen3Model : public basellm {
  public:
    std::string model_struct;

    std::string pre_prompt;
    std::string user_role;
    std::string bot_role;
    std::string history_sep;

    int max_positions;
    float rms_norm_eps;
    int rope_base;

    Qwen3Model();

    void InitParams();
};
