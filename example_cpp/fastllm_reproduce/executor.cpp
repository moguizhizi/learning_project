#include "executor.h"

#include <chrono>

#include "common_class.h"
#include "fastllm.h"
#include "file_utils.hpp"

void Executor::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    auto st = std::chrono::system_clock::now();
    bool lockInCPU = false;
    if (GetKVCacheInCPU() || GetHistoryCacheInCPU()) {
        // 暂时只有kvcache可能lock在CPU上
        for (auto &it : datas) {
            if (intParams.find(it.first + "___batch") != intParams.end()) {
                int batch = intParams.find(it.first + "___batch")->second;
                for (int i = 0; i < batch; i++) {
                    lockInCPU |= (((Data **)it.second)[i] && ((Data **)it.second)[i]->lockInCPU);
                }
            } else {
                lockInCPU |= (it.second && it.second->lockInCPU);
            }
        }
    }

    bool run = false;
    for (auto device : devices) {
        if (lockInCPU && device->deviceType != "cpu") {
            continue;
        }
        if (device->CanRun(opType, datas, floatParams, intParams)) {
#ifdef USE_CUDA
            if (device->deviceType == "cuda" && device->deviceIds.size() > 0) {
                FastllmCudaSetDevice(device->deviceIds[0]);
            }
            if (device->deviceType == "multicuda" && device->deviceIds.size() > 0) {
                FastllmMultiCudaSetDevice(device->deviceIds);
                if (device->deviceIdsRatio.size() > 0) {
                    FastllmMultiCudaSetDeviceRatio(device->deviceIdsRatio);
                }
            }
#endif
            bool intParamsSize = intParams.size();
            for (auto &it : datas) {
                if (intParamsSize > 0 && intParams.find(it.first + "___batch") != intParams.end()) {
                    int batch = intParams.find(it.first + "___batch")->second;
                    if ((it.first == "weights" || it.first == "biass") && ((Data **)it.second)[2]) {
                        if ((device->deviceType == "cpu" || device->deviceType == "numa" || device->deviceType == "tfacc") &&
                            ((Data **)it.second)[2]->dataDevice == DataDevice::CPU) {
                            continue;
                        }
                        if ((device->deviceType == "cuda" || device->deviceType == "multicuda") &&
                            ((Data **)it.second)[2]->dataDevice == DataDevice::CUDA) {
                            continue;
                        }
                    }
                    if ((it.first == "biass") && !((Data **)it.second)[2]) {
                        continue;
                    }
                    for (int i = 0; i < batch; i++) {
                        if (((Data **)it.second)[i]) {
                            ((Data **)it.second)[i]->ToDevice((void *)device);
                        }
                    }
                } else {
                    if (it.second) {
                        it.second->ToDevice((void *)device);
                    }
                }
            }
            device->Reshape(opType, datas, floatParams, intParams);
            device->Run(opType, datas, floatParams, intParams);
            run = true;
            break;
        }
    }
    if (!run) {
        ErrorInFastLLM("Can't run " + opType + " in any device.");
    }
    float spend = GetSpan(st, std::chrono::system_clock::now());
    profiler[opType] += spend;
}