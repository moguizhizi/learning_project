#pragma once // 或者用 #ifndef 方式

#include <cstdint> // C++ 推荐
#include <string>
#include <vector>

#include "common_struct.h"
#include "enum_space.h"

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

    bool multiDeviceData = false;
    std::map<int, Data *> multiDeviceDatas;

    uint64_t expansionSize = 0;
    uint64_t expansionBytes = 0;
    uint64_t cacheUid = 0;

    int unitSize = 0;
    int unitSizeDiv = 1;
    std::string name;

    std::string fileName;
    long long filePos;

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
    std::vector<void *> extraCudaData;
    std::vector<void *> extraCudaHalfData;

    std::vector<int> expansionDims;
    std::vector<uint64_t> strides;
    std::vector<int> dims;
    std::vector<int> dataDeviceIds;

    void UpdateUnitSize();
    void Resize(const std::vector<int> &dims);
    uint64_t Count(int i) const;
    uint64_t GetBytes() const;
    void Allocate();
    void Allocate(float v);
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
    void Reshape(const std::vector<int> &dims);
    void FakeFrom(const Data &ori, size_t offset);
};

class MoEQuantizedExecutor {
    Data **weights_;
    MoEQuantizedExecutor(Data **weights);

   private:
    std::vector<float> globalScales_;
    std::vector<float> globalZeros_;
    std::vector<uint8_t> globalInput_;
    std::vector<LowBitConfig> globalLowBitConfigs_;
    std::vector<float> globalSums_;
    std::vector<std::vector<float>> middles_;

   public:
    std::vector<std::vector<float>> results;

    void prepareBuffer(size_t n, size_t m, size_t group);
    void ensureMiddleAndResultBuffers(const std::vector<ExpertRoute> &routedExperts);
    void ExecuteForOuterIndex(int o, float *floatInput, int n, int m, const std::vector<ExpertRoute> &routedExperts, int permuteType);
};