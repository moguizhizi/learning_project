#pragma once

enum RoPEType { // 位置编码外推类型
    BASE = 0,
    LINEAR_SCALE = 1,
    STATIC_NTK = 2,
    DYMAMIC_NTK = 3,
    YARN = 4
};

enum DataDevice { CPU = 0, CUDA = 1 };

enum DataType {
    FLOAT32 = 0,
    BFLOAT16 = 1,
    INT16 = 2,
    INT8 = 3,
    INT4 = 4,
    INT2 = 5,
    BIT = 6,
    FLOAT16 = 7,
    INT4_NOZERO = 8, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale
    INT4_GROUP = 9,  // 不用zeroPoint的int4, floatValue = min + uint4Value * scale, 且使用分组量化
    FP8_E4M3 = 10,
    INT2_GROUP = 11,  // 不用zeroPoint的int2, floatValue = min + uint2Value * scale, 且使用分组量化
    BASE3_GROUP = 12, // 三元量化，-1 0 1
    INT32PARAM = 100, // int32的参数，这种类型的数据永远存在CPU上
    DATA_AUTO_NONE = 99999,
    DATA_AUTO_LINEAR,
    DATA_AUTO_EMBEDDING,
    DATA_AUTO_CONV
};

enum WeightType { NONE = 0, LINEAR = 1, EMBEDDING = 2, CONV2D = 3, AUTO = 99999 };