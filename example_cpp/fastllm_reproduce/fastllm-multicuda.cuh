#include "fastllm.h"

using DivisionScheme = std::map<int, std::vector<std::pair<int, int> > >;

std::vector<long long> FastllmCudaGetFreeSizes();

std::map<int, std::string> specialDeviceIds = {{99999, "cpu"}};

void SwitchDeviceAndGetInfos(int deviceId, std::string &specialId, int &mallocType);

void CopyToMultiDevices(Data &data, std::vector<int> devices, bool copyData);

void *AutoMalloc(size_t size, int type);

cudaError_t AutoMemset(void *a, int value, size_t size, int type);

cudaMemcpyKind GetCudaMemcpyType(int dstType, int srcType);

void FastllmMultiCudaSetDevice(std::vector<int> ids);

void FastllmMultiCudaSetDeviceRatio(std::map<int, int> &deviceRatio);

void FastllmGetMulticudaDeviceAndRatio(std::vector<int> &devices, std::map<int, int> &ratios, bool noSpecial);

std::vector<int> FastllmMultiCudaGetSplitPoints(
    std::vector<int> &multiCudaCurrentDevices, std::map<int, int> &multiCudaCurrentRatios, int total, int unit = 1);

cudaStream_t *GetFastllmStream(int id);

void EnablePeerAccessAll(const std::vector<int> &devices);

bool SplitMultiCudaWeight(Data &weight, Data &bias, std::vector<int> &multiCudaCurrentDevices, DivisionScheme divisionScheme, int splitAxis);