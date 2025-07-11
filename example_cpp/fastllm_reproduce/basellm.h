#pragma once

#include "enum_space.h"
#include "struct_space.hpp"
#include <string>

class Data {
  public:
    Data();
    Data(DataType datatype);
    Data(DataType datatype, const std::vector<int> &dims);
    Data(DataType datatype, const std::vector<int> &dims, DataDevice device, void *ptr);
    Data(DataType datatype, const std::vector<int> &dims, const std::vector<float> data);

    DataType dataType = DataType::FLOAT32;
    DataDevice dataDevice = DataDevice::CPU;

    bool isFake = false;
    bool directMemory = false;
    bool isKVCache = false;

    uint64_t expansionSize;
    uint64_t expansionBytes;
    uint64_t cacheUid;

    int unitSize;
    int unitSizeDiv;
    std::string name;

    uint8_t *cpuData = nullptr;
    void *cudaData = nullptr;

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
    void Expansion(const std::vector<int> &dims); // dims的格式[num_head, seqlen, head_dim]，且必须与原data的dims只保持seqlen的不同
    void ToDevice(DataDevice device);
    void ToDevice(DataDevice device, std::vector<int> &deviceIds);
    void CopyFrom(const Data &ori);
};

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

    Data sinData;
    Data cosData;

    void InitParams();
};
