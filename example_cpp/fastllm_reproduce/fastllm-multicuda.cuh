#include "fastllm.h"

std::vector<long long> FastllmCudaGetFreeSizes();

std::map<int, std::string> specialDeviceIds = {{99999, "cpu"}};

void SwitchDeviceAndGetInfos(int deviceId, std::string &specialId, int &mallocType);