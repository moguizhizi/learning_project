#pragma once

#include "alivethreadpool.h"

struct MultiThreadLinearBFloat16FP8E4M3Op : MultiThreadBaseOp {
    uint16_t *inputData;
    uint8_t *weightData;
    float *biasData, *outputData;
    int n, m, k, st, end;
    int blockK, blockM;
    float *scales;

    MultiThreadLinearBFloat16FP8E4M3Op(uint16_t *inputData,
                                       uint8_t *weightData,
                                       float *biasData,
                                       float *outputData,
                                       int n,
                                       int m,
                                       int k,
                                       int st,
                                       int end,
                                       float *scales,
                                       int blockK,
                                       int blockM);

    void Run();
};

struct MultiThreadLinearInt8Int4GroupOp : MultiThreadBaseOp {
    uint8_t *a, *b;
    float *c;
    int n, m, k, kstride;
    int *weightSums;
    float *weightMins;
    float *scales;
    float *bias;
    float *iscales, *izeros;
    float *inputSums;
    int group, groupCnt;

    MultiThreadLinearInt8Int4GroupOp(uint8_t *a,
                                     uint8_t *b,
                                     float *c,
                                     int n,
                                     int m,
                                     int k,
                                     int kstride,
                                     int *weightSums,
                                     float *weightMins,
                                     float *scales,
                                     float *bias,
                                     float *iscales,
                                     float *izeros,
                                     float *inputSums,
                                     int group,
                                     int groupCnt);

    void Run();
};

struct MultiThreadLinearFloat32Float32Op : MultiThreadBaseOp {
    float *inputData;
    float *weightData;
    float *biasData, *outputData;
    int n, m, k, st, end;

    MultiThreadLinearFloat32Float32Op(float *inputData, float *weightData, float *biasData, float *outputData, int n, int m, int k, int st, int end);

    void Run();
};

struct MultiThreadLinearFloat32Float16Op : MultiThreadBaseOp {
    float *inputData;
    uint16_t *weightData;
    float *biasData, *outputData;
    int n, m, k, st, end;

    MultiThreadLinearFloat32Float16Op(
        float *inputData, uint16_t *weightData, float *biasData, float *outputData, int n, int m, int k, int st, int end);

    void Run();
};