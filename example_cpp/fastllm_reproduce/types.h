#ifndef TYPES_H
#define TYPES_H

#include "common_class.h"
#include <map>
#include <string>

// 类型别名
typedef std::map<std::string, Data *> DataDict;
typedef std::map<std::string, float> FloatDict;
typedef std::map<std::string, int> IntDict;

#endif // TYPES_H
