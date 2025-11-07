#include "fastllm.h"

std::vector<long long> FastllmCudaGetFreeSizes();

std::map<int, std::string> specialDeviceIds = {{99999, "cpu"}};

void SwitchDeviceAndGetInfos(int deviceId, std::string &specialId, int &mallocType);

void CopyToMultiDevices(Data &data, std::vector<int> devices, bool copyData);

void *AutoMalloc(size_t size, int type);

cudaError_t AutoMemset(void *a, int value, size_t size, int type);

cudaMemcpyKind GetCudaMemcpyType(int dstType, int srcType);