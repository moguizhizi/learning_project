// safetensors.hpp

#pragma once

#include "json11.hpp"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct SafeTensorItem {
  std::string tensorName;
  std::string fileName;
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<int> intShape;
  std::vector<uint64_t> dataOffsets;

  uint64_t len = 1;
  uint64_t bytesLen = 1;

  SafeTensorItem();
  ~SafeTensorItem();
  SafeTensorItem(const std::string &tensorName, const std::string &fileName, uint64_t baseOffset, const json11::Json &config);
};

struct SafeTensors {
  std::map<std::string, SafeTensorItem> itmeDict;

  SafeTensors(const std::vector<std::string> fileNames);

  std::vector<std::string> GetSortedItemNames();
};
