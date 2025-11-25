#include "utils.h"

#include <cmath>
#include <cstring>

uint32_t as_uint(const float x) {
    return *(uint32_t *)&x;
}

float as_float(const uint32_t x) {
    return *(float *)&x;
}

float half_to_float(const uint16_t x) {         // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0,
                                                // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint32_t e = (x & 0x7C00) >> 10;      // exponent
    const uint32_t m = (x & 0x03FF) << 13;      // mantissa
    const uint32_t v = as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                    ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
}

uint16_t float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5,
                                        // +-5.9604645E-8, 3.311 digits
    const uint32_t b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint32_t e = (b & 0x7F800000) >> 23;  // exponent
    const uint32_t m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
           ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
           (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}

void Float16ToFloat32(uint16_t *float16, float *float32, int len) {
    int i = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i + 7 < len; i += 8) {
        float16x8_t input_vec = vld1q_f16((float16_t *)float16 + i);
        float32x4_t output_vec1 = vcvt_f32_f16(vget_low_f16(input_vec));
        float32x4_t output_vec2 = vcvt_f32_f16(vget_high_f16(input_vec));
        vst1q_f32(float32 + i, output_vec1);
        vst1q_f32(float32 + i + 4, output_vec2);
    }
#endif
    for (; i < len; i++) {
        float32[i] = g_fp16ToFp32Manager.dict[float16[i]];
    }
}

void Float32ToFloat16(float *float32, uint16_t *float16, int len) {
    int i = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i + 3 < len; i += 4) {
        float32x4_t input_vec = vld1q_f32(float32 + i);
        float16x4_t output_vec = vcvt_f16_f32(input_vec);
        vst1_f16((float16_t *)float16 + i, output_vec);
    }
#endif
#ifdef __AVX__
    for (; i + 7 < len; i += 8) {
        __m256 input_vec = _mm256_loadu_ps(float32 + i);                            // 加载 8 个 float32
        __m128i output_vec = _mm256_cvtps_ph(input_vec, _MM_FROUND_TO_NEAREST_INT); // 转换为 8 个 float16
        _mm_storeu_si128((__m128i *)(float16 + i), output_vec);                     // 存储 8 个 float16
    }
#endif
    for (; i < len; i++) {
        float16[i] = float_to_half(float32[i]);
    }
}

void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len) {
    int i = 0;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i + 3 < len; i += 4) {
        // Load 4 float32 values
        float32x4_t f32x4 = vld1q_f32(&float32[i]);

        // Reinterpret as uint32 to access bits
        uint32x4_t u32x4 = vreinterpretq_u32_f32(f32x4);

        // Shift right by 16 bits to get bfloat16 bits
        uint32x4_t shifted = vshrq_n_u32(u32x4, 16);

        // Narrow to 16-bit (takes bottom 16 bits from each 32-bit element)
        uint16x4_t bf16x4 = vmovn_u32(shifted);

        // Store 4 bfloat16 values
        vst1_u16(&bfloat16[i], bf16x4);
    }
#endif

#ifdef __AVX__
    for (; i + 7 < len; i += 8) {
        __m256i float_vec = _mm256_loadu_si256((__m256i *)&float32[i]);
        __m256i shifted = _mm256_srli_epi32(float_vec, 16);
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128((__m128i *)&bfloat16[i], packed);
    }
#endif
    // 标量处理剩余元素
    for (; i < len; i++) {
        uint32_t val;
        std::memcpy(&val, &float32[i], sizeof(val));
        bfloat16[i] = (uint16_t)(val >> 16);
    }
}

void Float16ToBFloat16(uint16_t *float16, uint16_t *bfloat16, int len) {
    int i = 0;
#ifdef __AVX__
    for (; i + 7 < len; i += 8) {
        __m256 _float_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(float16 + i)));
        __m256i float_vec = *((__m256i *)&_float_vec);
        __m256i shifted = _mm256_srli_epi32(float_vec, 16);
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128((__m128i *)&bfloat16[i], packed);
    }
#endif
    for (; i < len; i++) {
        uint32_t val;
        memcpy(&val, &g_fp16ToFp32Manager.dict[float16[i]], sizeof(val));
        bfloat16[i] = (uint16_t)(val >> 16);
    }
}

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1);
    return double(duration.count()) * std::chrono::nanoseconds::period::num / std::chrono::nanoseconds::period::den;
};

// ===== Manager 构造函数实现 =====
FP16ToFP32Manager::FP16ToFP32Manager() {
    for (int i = 0; i < 65536; i++) {
        dict[i] = half_to_float(i);
    }
}

BF16ToFP32Manager::BF16ToFP32Manager() {
    for (uint16_t i = 0; i < 65535; i++) {
        uint32_t x = (i << 16);
        dict[i] = *((float *)&x);
    }
}

BF16ToFP16Manager::BF16ToFP16Manager() {
    for (uint16_t i = 0; i < 65535; i++) {
        uint32_t x = (i << 16);
        dict[i] = float_to_half(*((float *)&x));
    }
}

FP16SiluManager::FP16SiluManager() {
    for (int i = 0; i < 65536; i++) {
        float x = half_to_float(i);
        float y = x / (1.0 + expf(-x));
        dict[i] = float_to_half(y);
    }
}

FP16SigmoidManager::FP16SigmoidManager() {
    for (int i = 0; i < 65536; i++) {
        float x = half_to_float(i);
        float y = 1.0 / (1.0 + expf(-x));
        dict[i] = float_to_half(y);
    }
}

// ===== 全局变量定义（只有 cpp 里有一次）=====
FP16ToFP32Manager g_fp16ToFp32Manager;
BF16ToFP32Manager g_bf16tofp32;
BF16ToFP16Manager g_bf16tofp16;
FP16SiluManager g_fp16SiluManager;
FP16SigmoidManager g_fp16SigmoidManager;
