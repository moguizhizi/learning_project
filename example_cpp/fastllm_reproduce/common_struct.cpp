#include "common_struct.h"
#include <algorithm>
#include <cmath>

LowBitConfig::LowBitConfig() {}

LowBitConfig::LowBitConfig(float max, float min, int type, uint8_t bit) {
    this->max = max;
    this->min = min;
    this->type = type; // 0: 有zero点 1: 不需要zero点
    this->bit = bit;
}

void LowBitConfig::Reset() {
    this->min = std::min(this->min, 0.f);
    this->max = std::max(this->max, 0.f);

    const uint8_t qmin = 0;
    const uint8_t qmax = (1 << this->bit) - 1;

    this->scale = (this->max - this->min) / (qmax - qmin);
    const float initial_zero_point = qmin - (this->min / this->scale);

    if (initial_zero_point < qmin) {
        this->zeroPoint = qmin;
    } else if (initial_zero_point > qmax) {
        this->zeroPoint = qmax;
    } else {
        this->zeroPoint = static_cast<u_int8_t>(std::round(initial_zero_point));
    }

    if (type == 1) {
        this->min = -this->scale * zeroPoint;
        return;
    }
}

uint8_t LowBitConfig::quantization(const float &realNumber) const {
    if (this->type == 0) {
        return (uint8_t)(std::min((double)((1 << bit) - 1), (double)std::max(realNumber / this->scale + this->zeroPoint + 0.5, 0.0)));
    } else {
        return (uint8_t)(std::max(0.f, std::min(15.f, (realNumber - this->min) / this->scale + 0.5f)));
    }
}

float LowBitConfig::invQuantization(const uint8_t &qNumber) const {
    if (this->type == 0) {
        return (this->scale * ((float)qNumber - (float)this->zeroPoint));
    } else {
        return this->min + this->scale * qNumber;
    }
}