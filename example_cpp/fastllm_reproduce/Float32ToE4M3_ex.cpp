// FP32 量化为 FP8 (E4M3)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

// 转换 FP32 为 8 位浮点数 (E4M3)
// FP8 格式: 1 位符号 | 4 位指数 | 3 位尺缘
uint8_t Float32ToE4M3(float f32) {
    if (f32 == 0.0f)
        return 0; // 零处理

    union {
        float f;
        uint32_t u;
    } u = {f32};

    uint32_t sign = (u.u >> 31) & 0x1;
    int32_t exponent = ((u.u >> 23) & 0xFF) - 127; // FP32 真实指数
    uint32_t mantissa = u.u & 0x7FFFFF;            // 23 位尺缘

    // FP8 E4M3 参数
    int fp8_bias = 8;
    int fp8_exp = exponent + fp8_bias;

    // 超出指数范围
    if (fp8_exp >= 0xF) { // 最大 15
        fp8_exp = 0xF;
        mantissa = 0x7; // max mantissa
    } else if (fp8_exp <= 0) {
        // 指数太小，转为 0
        return 0;
    }

    // 截断尺缘 (23-bit -> 3-bit)
    uint8_t fp8_mantissa = (mantissa >> (23 - 3)) & 0x7;

    // 组合 FP8
    uint8_t fp8 = (sign << 7) | ((fp8_exp & 0xF) << 3) | fp8_mantissa;
    return fp8;
}

// 清晰显示 FP8 (E4M3) 为三元组
void PrintFp8(uint8_t fp8) {
    uint8_t sign = (fp8 >> 7) & 0x1;
    uint8_t exponent = (fp8 >> 3) & 0xF;
    uint8_t mantissa = fp8 & 0x7;

    printf("FP8 = sign:%u exp:%u mant:%u\n", sign, exponent, mantissa);
}

int main() {
    float test_vals[] = {0.0f, 0.125f, -1.5f, 3.14f, 100.0f, -0.0001f};

    for (float val : test_vals) {
        uint8_t fp8 = Float32ToE4M3(val);
        printf("\n[%.6f] -> FP8 = 0x%02X\n", val, fp8);
        PrintFp8(fp8);
    }

    return 0;
}
