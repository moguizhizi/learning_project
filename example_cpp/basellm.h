#include "struct_space.hpp"
#include <string>

class basellm {
  public:
    basellm();
    ~basellm();

    std::string model_type;
    std::string model_struct;

    std::string pre_prompt;
    std::string user_role;
    std::string bot_role;
    std::string history_sep;

    int block_cnt = 28;
    int rotary_dim = 64;

    WeightMap weight;
};