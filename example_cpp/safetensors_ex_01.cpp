#ifndef __FILE_defined
#define __FILE_defined 1

struct _IO_FILE;

/* The opaque type of streams.  This is the definition used elsewhere.  */
typedef struct _IO_FILE FILE;

#endif

#include <string>
#include <set>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include "json11.hpp"
#include <iostream>

struct SafeTensors
{

    SafeTensors(const std::vector<std::string> fileNames)
    {
        for (const std::string &fileName : fileNames)
        {
            FILE *file = fopen(fileName.c_str(), "rb");
            if (!file)
            {
                fprintf(stderr, "[Line %d] Failed to open file: %s\n", __LINE__, fileName.c_str());
                exit(0);
            }
            uint64_t stlen;
            int ret = fread(&stlen, sizeof(uint64_t), 1, file);
            if (ret != 1)
            {
                fprintf(stderr, "[Line %d] Failed read from : %s\n", __LINE__, fileName.c_str());
                fclose(file);
                exit(0);
            }

            char *layers_info = new char[stlen + 5];
            layers_info[stlen] = 0;
            ret = fread(layers_info, 1, stlen, file);
            if (ret != stlen)
            {
                fprintf(stderr, "[Line %d] Failed read from : %s\n", __LINE__, fileName.c_str());
                fclose(file);
                exit(0);
            }
            std::string error;
            auto config = json11::Json::parse(layers_info, error);
            for (auto &it : config.object_items())
                std::cout << it.first << ":" << it.second.dump() << std::endl;
        }
    }
};

int main()
{
    std::vector<std::string> fileNames = {"/home/temp/llm_model/Qwen/Qwen2.5-VL-7B-Instruct/model-00004-of-00005.safetensors"};
    SafeTensors safetensors(fileNames);
}