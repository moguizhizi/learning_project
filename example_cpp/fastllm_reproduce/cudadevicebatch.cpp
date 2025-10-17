#include "cudadevicebatch.h"
#include "fastllm-cuda.cuh"
#include "file_utils.hpp"

void DoCudaAttentionBatch(Data **qs, Data **ks, Data **vs, Data **masks, Data **outputs, int group, float scale, int batch) {
    long long aveLen = 0;
    for (int i = 0; i < batch; i++) {
        aveLen += ks[i]->dims[1];
    }
    aveLen /= batch;
    if (qs[0]->dataType == DataType::FLOAT32 || true) {
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
        }
        FastllmCudaAttentionBatch(qs, ks, vs, masks, outputs, group, scale, batch);
    } else {
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
        }
        for (int i = 0; i < batch; i++) {
            if (qs[i]->dataType == DataType::FLOAT16) {
                if (masks == nullptr || masks[i] == nullptr) {
                    FastllmCudaHalfAttention(*qs[i], *ks[i], *vs[i], Data(), *outputs[i], group, scale, 0);
                } else {
                    FastllmCudaHalfAttention(*qs[i], *ks[i], *vs[i], *masks[i], *outputs[i], group, scale, 0);
                }
            } else {
                ErrorInFastLLM("AttentionBatch: datatype should be float32 or float16.");
            }
        }
    }
}