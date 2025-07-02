#ifndef __FILE_defined
#define __FILE_defined 1

struct _IO_FILE;

/* The opaque type of streams.  This is the definition used elsewhere.  */
typedef struct _IO_FILE FILE;

#endif

#include "json11.hpp"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

struct SafeTensorItem {
  std::string tensorName;
  std::string fileName;
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint64_t> intShape;
  std::vector<uint64_t> dataOffsets;

  uint64_t len = 1;
  uint64_t bytesLen = 1;

  SafeTensorItem() {};
  ~SafeTensorItem() {

  };
  SafeTensorItem(const std::string &tensorName, const std::string &fileName, uint64_t baseOffset, const json11::Json &config) {
    this->tensorName = tensorName;
    this->fileName = fileName;
    this->dtype = config["dtype"].string_value();

    for (auto &it : config["shape"].array_items()) {
      this->shape.push_back(it.ll_value());
    }

    for (auto &it : config["data_offsets"].array_items()) {
      this->dataOffsets.push_back(baseOffset + it.ll_value());
    }

    for (auto &it : this->shape) {
      this->len = this->len * it;
    }

    this->bytesLen = this->dataOffsets[1] - this->dataOffsets[0];
  };
};

struct SafeTensors {
  std::map<std::string, SafeTensorItem> itmeDict;

  SafeTensors(const std::vector<std::string> fileNames) {
    for (const std::string &fileName : fileNames) {
      FILE *file = fopen(fileName.c_str(), "rb");
      if (!file) {
        fprintf(stderr, "[Line %d] Failed to open file: %s\n", __LINE__, fileName.c_str());
        exit(0);
      }
      uint64_t stlen;
      int ret = fread(&stlen, sizeof(uint64_t), 1, file);
      if (ret != 1) {
        fprintf(stderr, "[Line %d] Failed read from : %s\n", __LINE__, fileName.c_str());
        fclose(file);
        exit(0);
      }

      char *layers_info = new char[stlen + 5];
      layers_info[stlen] = 0;
      ret = fread(layers_info, 1, stlen, file);
      if (ret != stlen) {
        fprintf(stderr, "[Line %d] Failed read from : %s\n", __LINE__, fileName.c_str());
        fclose(file);
        exit(0);
      }
      std::string error;
      auto config = json11::Json::parse(layers_info, error);
      for (auto &it : config.object_items()) {
        // std::cout << it.first << ":" << it.second.dump() << std::endl;
        if (it.first != "__metadata__") {
          std::cout << it.first << ":" << it.second.dump() << std::endl;
          this->itmeDict[it.first] = SafeTensorItem(it.first, fileName, stlen + 8, it.second);
        }
      }
      delete[] layers_info;
    }
  }
};

int main() {
  std::vector<std::string> fileNames = {"/home/temp/llm_model/Qwen/Qwen2.5-VL-7B-Instruct/"
                                        "model-00004-of-00005.safetensors"};
  SafeTensors safetensors(fileNames);
}