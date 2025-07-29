// safetensors.hpp

#pragma once

#include "enum_space.h"
#include "json11.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct SafeTensorItem {
    std::string tensorName;
    std::string fileName;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<int> intShape;
    std::vector<uint64_t> dataOffsets;

    uint64_t len = 1;
    uint64_t bytes = 1;
    int blockK;
    int blockM;

    uint8_t *buffer = nullptr;
    float *minsBuffer = nullptr, *scalesBuffer = nullptr;

    SafeTensorItem();
    ~SafeTensorItem();
    SafeTensorItem(const std::string &tensorName, const std::string &fileName, uint64_t baseOffset, const json11::Json &config);
    void ClearBuffer();
    void CreateBuffer(DataType dstType);
    void CreateBufferWithScale(DataType dstType, SafeTensorItem &scale);
    void CreateBufferWithAWQ(DataType dstType, SafeTensorItem &scale, SafeTensorItem &qzero);
    void Transpose(DataType type);
};

struct SafeTensors {
    std::map<std::string, SafeTensorItem> itmeDict;
    std::set<std::string> fileNames;

    SafeTensors(const std::set<std::string> fileNames);

    std::vector<std::string> GetSortedItemNames();
};

struct Tokenizer {};

struct WeightMap {
    std::set<std::string> embeddingsNames;
    std::set<std::string> linearNames;
    std::map<std::string, std::string> dicts;

    void AddDict(const std::string &key, const std::string &value);
    WeightType GetWeightType(const std::string &key);
};

struct WeightMergeRuleSingle {
    std::vector<std::string> inputs;
    std::string output;
    std::string type;

    WeightMergeRuleSingle(const std::vector<std::string> &inputs, std::string output, std::string type);
};

struct WeightMergeRule {
    std::vector<WeightMergeRuleSingle> rules;
    std::set<std::string> allInputs;

    WeightMergeRule(const std::vector<WeightMergeRuleSingle> &rules);
};

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer();
    CudaMemoryBuffer(void *data, size_t size, bool busy);
};

struct FP8E4M3ToFP32Manager {
    float dict[256] = {
        0.0,         0.001953125,  0.00390625,  0.005859375,  0.0078125,   0.009765625,  0.01171875, 0.013671875,  0.015625,    0.017578125,
        0.01953125,  0.021484375,  0.0234375,   0.025390625,  0.02734375,  0.029296875,  0.03125,    0.03515625,   0.0390625,   0.04296875,
        0.046875,    0.05078125,   0.0546875,   0.05859375,   0.0625,      0.0703125,    0.078125,   0.0859375,    0.09375,     0.1015625,
        0.109375,    0.1171875,    0.125,       0.140625,     0.15625,     0.171875,     0.1875,     0.203125,     0.21875,     0.234375,
        0.25,        0.28125,      0.3125,      0.34375,      0.375,       0.40625,      0.4375,     0.46875,      0.5,         0.5625,
        0.625,       0.6875,       0.75,        0.8125,       0.875,       0.9375,       1.0,        1.125,        1.25,        1.375,
        1.5,         1.625,        1.75,        1.875,        2.0,         2.25,         2.5,        2.75,         3.0,         3.25,
        3.5,         3.75,         4.0,         4.5,          5.0,         5.5,          6.0,        6.5,          7.0,         7.5,
        8.0,         9.0,          10.0,        11.0,         12.0,        13.0,         14.0,       15.0,         16.0,        18.0,
        20.0,        22.0,         24.0,        26.0,         28.0,        30.0,         32.0,       36.0,         40.0,        44.0,
        48.0,        52.0,         56.0,        60.0,         64.0,        72.0,         80.0,       88.0,         96.0,        104.0,
        112.0,       120.0,        128.0,       144.0,        160.0,       176.0,        192.0,      208.0,        224.0,       240.0,
        256.0,       288.0,        320.0,       352.0,        384.0,       416.0,        448.0,      480,          -0.0,        -0.001953125,
        -0.00390625, -0.005859375, -0.0078125,  -0.009765625, -0.01171875, -0.013671875, -0.015625,  -0.017578125, -0.01953125, -0.021484375,
        -0.0234375,  -0.025390625, -0.02734375, -0.029296875, -0.03125,    -0.03515625,  -0.0390625, -0.04296875,  -0.046875,   -0.05078125,
        -0.0546875,  -0.05859375,  -0.0625,     -0.0703125,   -0.078125,   -0.0859375,   -0.09375,   -0.1015625,   -0.109375,   -0.1171875,
        -0.125,      -0.140625,    -0.15625,    -0.171875,    -0.1875,     -0.203125,    -0.21875,   -0.234375,    -0.25,       -0.28125,
        -0.3125,     -0.34375,     -0.375,      -0.40625,     -0.4375,     -0.46875,     -0.5,       -0.5625,      -0.625,      -0.6875,
        -0.75,       -0.8125,      -0.875,      -0.9375,      -1.0,        -1.125,       -1.25,      -1.375,       -1.5,        -1.625,
        -1.75,       -1.875,       -2.0,        -2.25,        -2.5,        -2.75,        -3.0,       -3.25,        -3.5,        -3.75,
        -4.0,        -4.5,         -5.0,        -5.5,         -6.0,        -6.5,         -7.0,       -7.5,         -8.0,        -9.0,
        -10.0,       -11.0,        -12.0,       -13.0,        -14.0,       -15.0,        -16.0,      -18.0,        -20.0,       -22.0,
        -24.0,       -26.0,        -28.0,       -30.0,        -32.0,       -36.0,        -40.0,      -44.0,        -48.0,       -52.0,
        -56.0,       -60.0,        -64.0,       -72.0,        -80.0,       -88.0,        -96.0,      -104.0,       -112.0,      -120.0,
        -128.0,      -144.0,       -160.0,      -176.0,       -192.0,      -208.0,       -224.0,     -240.0,       -256.0,      -288.0,
        -320.0,      -352.0,       -384.0,      -416.0,       -448.0,      -480};
};

struct BF16ToFP32Manager {
    float dict[65536];

    BF16ToFP32Manager();
};

struct BF16ToFP16Manager {
    float dict[65536];

    BF16ToFP16Manager();
};

struct LowBitConfig {
    float max;
    float min;
    int type;
    int bit;
    uint8_t zeroPoint;
    float scale;

    LowBitConfig();
    LowBitConfig(float max, float min, int type, uint8_t bit);

    void Reset();
    uint8_t quantization(const float &realNumber) const;
    float invQuantization(const uint8_t &qNumber) const;
};

struct MultiThreadBaseOp {
    virtual void Run() = 0;
};

struct MultiThreadGroupQuantizationOp : MultiThreadBaseOp {
    int st, end, m;
    float *f;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int group, groupCnt;

    MultiThreadGroupQuantizationOp(int st, int end, int m, int bit, LowBitConfig *configs, int group, int groupCnt, float *f, uint8_t *u8);
    void Run() override;
};

struct MultiThreadGroupQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int group, groupCnt;

    MultiThreadGroupQuantizationBF16Op(int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int group, int groupCnt);
    void Run() override;
};

struct MultiThreadPerChannelQuantizationOp : MultiThreadBaseOp {
    int st, end, m;
    float *f;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;

    MultiThreadPerChannelQuantizationOp(int st, int end, int m, float *f, uint8_t *u8, LowBitConfig *configs, int bit);
    void Run() override;
};

struct MultiThreadPerChannelQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;

    MultiThreadPerChannelQuantizationBF16Op(int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit);
    void Run() override;
};

struct MultiThreadBase3GroupQuantizationOp : MultiThreadBaseOp {
    int st, end, m;
    float *f32;
    uint8_t *u8;
    uint16_t *halfScales;
    int group;
    int groupCnt;

    MultiThreadBase3GroupQuantizationOp(int st, int end, int m, float *f32, uint8_t *u8, uint16_t *scale, int group, int groupCnt);
    void Run() override;
};

struct MultiThreadBase3GroupQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    uint16_t *halfScales;
    int group, groupCnt;

    MultiThreadBase3GroupQuantizationBF16Op(int st, int end, int m, uint16_t *bf, uint8_t *u8, uint16_t *halfScales, int group, int groupCnt);
    void Run() override;
};