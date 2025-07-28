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

    // 以下参数用于量化，对FLOAT数据不适用
    int perChannelAxis = -1;       // 沿哪个轴分通道量化，-1代表没有分通道
    int group = -1, groupCnt = -1; // 分组量化，group代表组数，groupCnt代表每组有多少个元素，-1代表不使用分组量化

    // FP8的分组量化， [blockK, blockM]的小矩阵为一组
    int blockK = -1, blockM = -1;

    uint8_t *cpuData = nullptr;
    void *cudaData = nullptr;

    std::vector<int> expansionDims;
    std::vector<uint64_t> stride;
    std::vector<int> dims;
    std::vector<int> dataDeviceIds;

    std::vector<float> scales, mins;

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
    void CreateFromOriData(
        WeightType weightType, DataType oriDataType, uint8_t *oriData, float *oriMins, float *oriScales, int groupCnt, int blockK, int blockM);
};

class basellm {
  public:
    basellm();
    ~basellm();

    std::string model_type;
    std::set<std::string> cantQuantLinears;
    std::set<std::string> moelinears;

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
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> GetTensorMap(const std::vector<std::string> &tensorNames);
};
