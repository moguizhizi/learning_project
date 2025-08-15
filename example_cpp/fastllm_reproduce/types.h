#ifndef TYPES_H
#define TYPES_H

#include <map>
#include <string>

// 前向声明 Data
class Data;

// 类型别名
typedef std::map<std::string, Data *> DataDict;
typedef std::map<std::string, float> FloatDict;
typedef std::map<std::string, int> IntDict;

#endif // TYPES_H
