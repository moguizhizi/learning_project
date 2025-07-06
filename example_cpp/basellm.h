#pragma once

#include "enum_space.h"
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

class Data {
  public:
    Data(DataType datatype);
    Data(DataType datatype, const std::vector<int> &dims);
    Data(DataType datatype, const std::vector<int> &dims, DataDevice device, void *ptr);
    Data(DataType datatype, const std::vector<int> &dims, DataDevice device, const std::vector<float> data);

    DataType dataType = DataType::FLOAT32;
    DataDevice dataDevice = DataDevice::CPU;

    bool isFake = false;

    uint64_t expansionSize;
    uint64_t expansionBytes;

    int unitSize;
    int unitSizeDiv;

    uint8_t *cpudata = nullptr;
    void *cudadata = nullptr;

    std::vector<int> expansionDims;
    std::vector<uint64_t> stride;
    std::vector<int> dims;
    std::vector<int> dataDeviceIds;

    void UpdateUnitSize();
    void Resize(const std::vector<int> &dims);
    uint64_t Count(int i) const;
    uint64_t GetBytes() const;
    void Allocate();
    void FreeSpace();
    void MallocSpace(uint64_t size_t);
};