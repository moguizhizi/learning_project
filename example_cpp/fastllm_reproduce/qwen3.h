#pragma once

#include "basellm.h"
#include "enum_space.h"

class Qwen3Model : public basellm {
  public:
    std::string model_struct;

    std::string pre_prompt;
    std::string user_role;
    std::string bot_role;
    std::string history_sep;

    int max_positions;
    float rms_norm_eps;

    Qwen3Model();

    void InitParams();
    std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float rope_base, float rope_factor, int seqlen);
    void WarmUp();

  protected:
    float rope_base = 10000.f;
    RoPEType rope_type = RoPEType::BASE;
    float rope_scale = 1.0;
    std::vector<std::vector<float>> sin;
    std::vector<std::vector<float>> cos;

    float rope_factor = 1.0;
};
