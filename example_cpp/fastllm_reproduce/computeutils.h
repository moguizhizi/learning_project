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

void RunLinearFloat32Float32(
    float *inputData, float *weightData, float *outputData, float *biasData, int n, int m, int k, AliveThreadPool *pool, int startTid, int threadNum);

void RunLinearInt8Int4Group(uint8_t *a,
                            uint8_t *b,
                            float *c,
                            int n,
                            int m,
                            int k,
                            int group,
                            int groupCnt,
                            int *weightSums,
                            float *weightMins,
                            float *scales,
                            float *bias,
                            float *inputSums,
                            float *iscales,
                            float *izeros,
                            AliveThreadPool *pool,
                            int startTid,
                            int threadNum);

void RunLinearFloat32Int4Group(float *inputData,
                               Data &weight,
                               float *outputData,
                               float *biasData,
                               int n,
                               int m,
                               int k,
                               int group,
                               int groupCnt,
                               AliveThreadPool *pool,
                               int startTid,
                               int threadNum);

void RunLinearFloat32Float16(float *inputData,
                             uint16_t *weightData,
                             float *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum);

void MatMulInt8Int8(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride);

struct MultiThreadLinearInt8Int8Op : MultiThreadBaseOp {
    uint8_t *a;
    uint8_t *b;
    int32_t *c;
    int n, m, k, kstride;
    int *weightSums, *weightZeros;
    float *scales, *bias;
    float *iscales, *izeros, *inputSums;

    MultiThreadLinearInt8Int8Op(uint8_t *a,
                                uint8_t *b,
                                int32_t *c,
                                int n,
                                int m,
                                int k,
                                int kstride,
                                int *weightSums,
                                int *weightZeros,
                                float *scales,
                                float *bias,
                                float *iscales,
                                float *izeros,
                                float *inputSums);

    void Run();
};

struct MultiThreadLinearFloat32Int2GroupOp : MultiThreadBaseOp {
    float *inputData;
    Data *weight;
    float *biasData, *outputData;
    int n, m, k, st, end;

    MultiThreadLinearFloat32Int2GroupOp(float *inputData, Data *weight, float *biasData, float *outputData, int n, int m, int k, int st, int end);

    void Run();
};

void RunLinearFloat32Int2Group(float *inputData,
                               Data &weight,
                               float *outputData,
                               float *biasData,
                               int n,
                               int m,
                               int k,
                               int group,
                               int groupCnt,
                               AliveThreadPool *pool,
                               int startTid,
                               int threadNum);

void RunLinearFloat16Float32(uint16_t *inputData,
                             float *weightData,
                             uint16_t *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum);

struct MultiThreadLinearFloat16Float16Op : MultiThreadBaseOp {
    uint16_t *inputData;
    uint16_t *weightData;
    float *biasData;
    uint16_t *outputData;
    int n, m, k, st, end;

    MultiThreadLinearFloat16Float16Op(
        uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, int n, int m, int k, int st, int end);

    void Run();
};

void MatMulFloat16Float16(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, int n, int m, int k, int st, int end);

void RunLinearFloat16Float16(uint16_t *inputData,
                             uint16_t *weightData,
                             uint16_t *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum);

void RunLinearInt8Int8(uint8_t *a,
                       uint8_t *b,
                       float *c,
                       int n,
                       int m,
                       int k,
                       int *weightSums,
                       int *weightZeros,
                       float *scales,
                       float *bias,
                       float *inputSums,
                       float *iscales,
                       float *izeros,
                       AliveThreadPool *pool,
                       int startTid,
                       int threadNum);

void RunLinearFloat32Int8(
    float *inputData, Data &weight, float *outputData, float *biasData, int n, int m, int k, AliveThreadPool *pool, int startTid, int threadNum);

void RunLinearFloat16Int4Group(uint16_t *inputData,
                               Data &weight,
                               uint16_t *outputData,
                               float *biasData,
                               int n,
                               int m,
                               int k,
                               int group,
                               int groupCnt,
                               AliveThreadPool *pool,
                               int startTid,
                               int threadNum);

void RunLinearFloat16FP8E4M3(uint16_t *inputData,
                             Data &weight,
                             uint16_t *outputData,
                             float *biasData,
                             int n,
                             int m,
                             int k,
                             AliveThreadPool *pool,
                             int startTid,
                             int threadNum);