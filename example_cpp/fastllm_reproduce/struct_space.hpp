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
#include <unicode/unistr.h>
#include <unordered_map>
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

struct Tokenizer {
    enum TokenizerType { BPE = 0, NORMAL = 1, QWEN = 2, GLM = 3, BERT = 4 };

    struct TrieNode {
        int tokenId;
        float score;
        std::map<int, TrieNode *> next;
        TrieNode();
    };

    json11::Json tokenizerConfig;
    std::string chatTemplate = "";

    bool addDummyPrefix = true;         // 是否在首位添加空格
    bool removeExtraWhitespaces = true; // 是否将多个空格合并为一个
    bool byteAsChar = false;            // 是否将byte变为展示字符

    std::unordered_map<int, std::string> tokenToStringDict;
    std::unordered_map<int, float> tokenToScoreDict;
    std::unordered_map<std::string, int> stringToTokenDict;
    std::vector<std::string> specialTokens;

    std::unordered_map<wchar_t, wchar_t> byteCharDict;
    std::unordered_map<wchar_t, wchar_t> charByteDict;

    TrieNode *root = nullptr;
    TrieNode *specialRoot = nullptr;

    TokenizerType type = TokenizerType::BPE;

    Tokenizer();
    void SetTokenizerConfig(const json11::Json &config);
    void SetChatTemplate();
    void Insert(const std::string &s, int tokenId, float score = 1.0f); // 插入一个token
    std::string Normalize(const std::string &ori, const bool addDummyPrefix = true);
    void SetSpecialTokens(const std::map<std::string, int> &specialTokenMap);
    std::string WstringToUtf8(const std::wstring &wstr);
    std::wstring Utf8ToWstring(const std::string &utf8Str);
    int GetTokenId(const std::string &s);
};

struct WeightMap {
    Tokenizer tokenizer;

    std::set<std::string> embeddingsNames;
    std::set<std::string> linearNames;
    std::map<std::string, std::string> dicts;
    std::map<std::string, Data> weight;

    void AddDict(const std::string &key, const std::string &value);
    void AddTokenizerWord(const std::string &key, int value, float score);
    void AddEmptyWeight(const std::string &key, const std::vector<int> &dims, DataType dataType);
    WeightType GetWeightType(const std::string &key);
    Data &operator[](const std::string &key);
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
    int type;

    MultiThreadGroupQuantizationOp(int st, int end, int m, int bit, LowBitConfig *configs, int group, int groupCnt, float *f, uint8_t *u8, int type);
    void Run() override;
};

struct MultiThreadGroupQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int group, groupCnt;
    int type;

    MultiThreadGroupQuantizationBF16Op(
        int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int group, int groupCnt, int type);
    void Run() override;
};

struct MultiThreadPerChannelQuantizationOp : MultiThreadBaseOp {
    int st, end, m;
    float *f;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int type;

    MultiThreadPerChannelQuantizationOp(int st, int end, int m, float *f, uint8_t *u8, LowBitConfig *configs, int bit, int type);
    void Run() override;
};

struct MultiThreadPerChannelQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int type;

    MultiThreadPerChannelQuantizationBF16Op(int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int type);
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

struct ByteWriter {
    uint8_t *cur;

    ByteWriter(uint8_t *data);

    void WriteInt(int v);
    void WriteFloat(float v);
    void WriteString(const std::string &s);
    void WriteBytes(uint8_t *buffer, uint64_t bytes);
};

struct ByteReader {
    uint8_t *cur;

    ByteReader(uint8_t *data);
    int ReadInt();
    float ReadFloat();
    std::string ReadString();
    void ReadBytes(uint8_t *buffer, uint64_t bytes);
};

struct ComputeServer {
    volatile uint8_t *baseAddr;
    volatile uint8_t *baseOutputAddr;
    volatile int *flag;

    std::vector<uint8_t> inputBuffer;
    std::vector<uint8_t> outputBuffer;

    ComputeServer(int partId, int partCnt, int threadNum);
    void Start();
};

struct NumaClient {
    int serverNumaCnt;

    volatile int32_t *flag;
    volatile uint8_t *result;
    volatile uint8_t *buf;

    NumaClient();

    void Launch(int opType);
    void Wait();
};

struct MultiThreadBaseOp {
    virtual void Run() = 0;
};

struct MultiThreadSingleAttentionOp : MultiThreadBaseOp {
    float *qd, *kd, *vd, *maskd, *od;
    float scale;
    int q1, q2, k1, v2;

    MultiThreadSingleAttentionOp(float *qd, float *kd, float *vd, float *maskd, float *od, float scale, int q1, int q2, int k1, int v2);
    void Run();
};

struct MultiThreadSingleAttentionFloat16Op : MultiThreadBaseOp {
    uint16_t *qd, *kd, *vd, *maskd, *od;
    float scale;
    int q1, q2, k1, v2;
    MultiThreadSingleAttentionFloat16Op(
        uint16_t *qd, uint16_t *kd, uint16_t *vd, uint16_t *maskd, uint16_t *od, float scale, int q1, int q2, int k1, int v2);

    void Run();
};