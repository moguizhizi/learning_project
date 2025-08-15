#pragma once

#include "enum_space.h"
#include "struct_space.hpp"
#include "types.h"
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
    WeightType weightType = WeightType::NONE;

    bool isFake = false;
    bool directMemory = false;
    bool isKVCache = false;

    uint64_t expansionSize = 0;
    uint64_t expansionBytes = 0;
    uint64_t cacheUid = 0;

    int unitSize = 0;
    int unitSizeDiv = 1;
    std::string name;

    // 以下参数用于量化，对FLOAT数据不适用
    int perChannelAxis = -1;       // 沿哪个轴分通道量化，-1代表没有分通道
    int group = -1, groupCnt = -1; // 分组量化，group代表组数，groupCnt代表每组有多少个元素，-1代表不使用分组量化

    // 以下为每个通道/分组的量化参数
    // 1. 若不使用分通道量化，那么总组数 = 1
    // 2. 若使用分通道量化，那么总组数 = 通道数
    // 3. 若使用分组量化，那么总组数 = 通道数 * 组数(group)
    std::vector<LowBitConfig>
        perChannelsConfigs; // perChannelsConfigs[i]代表第i个通道的min, max; 如果没有分通道，perChannelsConfigs[0]代表全局min, max
    std::vector<float> scales, mins;
    std::vector<int> zeros;
    std::vector<int> weightSum; // 作为权重时，有时候需要存一些和加速计算

    std::vector<uint16_t> halfScales;

    // FP8的分组量化， [blockK, blockM]的小矩阵为一组
    int blockK = -1, blockM = -1;

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
    void CreateFromOriData(
        WeightType weightType, DataType oriDataType, uint8_t *oriData, float *oriMins, float *oriScales, int groupCnt, int blockK, int blockM);
    void ExportFastllmFormat(uint8_t *bytes);
    uint64_t GetFastllmFormateBytes();
    void CreateFromFastllmFormat(uint8_t *datas, uint64_t len);
    void CalcWeightSum();
};

class basellm {
  public:
    basellm();
    ~basellm();

    virtual void WarmUp() {}; // 预热

    std::string model_type;
    std::set<std::string> cantQuantLinears;
    std::set<std::string> moeLinears;
    std::set<int> eos_token_ids;

    int block_cnt;
    int rotary_dim;
    int head_dim = 0;

    std::string pre_prompt;                       // 最初对话的提示语
    std::string user_role, bot_role, history_sep; // 用于生成每一轮的prompt

    int bos_token_id;
    int eos_token_id;

    int embed_dim;
    int num_attention_heads = 0;

    int num_key_value_heads;

    WeightMap weight;

    std::vector<WeightMergeRule> weightMergeRules;
    std::map<std::string, std::string> specialWeights; // 一些特殊层，可以提前注册（一般用于TFACC）

    Data sinData;
    Data cosData;

    void InitParams();
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> GetTensorMap(const std::vector<std::string> &tensorNames);
    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    basellm::GetTensorMap(const std::vector<std::string> &tensorNames, bool useMoeDataType, DataType moeDataType);
    void MergeWeightsFromRules(const std::string &weightName,
                               const std::set<std::string> &allWeightNames,
                               const std::set<std::string> &allFinishName,
                               bool &needMerge);
};

class BaseDevice {};
class CpuDevice : BaseDevice {};
class BaseOperator {
  public:
    // 是否可以运行某一个算子
    virtual bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

    // 对某一个算子进行形状推理
    virtual void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

    // 对某一个算子进行推理
    virtual void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) = 0;
};

class CpuToFloat16 : BaseOperator {
    bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};