// common_struct.h
#pragma once // 或者用 #ifndef 方式

#include <cstdint> // C++ 推荐

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