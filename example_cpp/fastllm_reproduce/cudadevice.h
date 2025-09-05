#pragma once // 或者用 #ifndef 方式

#include "device.h"

void DoCudaLinearReshape(Data &input, Data &weight, Data &output);