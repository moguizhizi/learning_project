#include "struct_space.hpp"

int main() {
    std::string lorapath = "/home/temp";
    std::string path = lorapath;

    if (path.back() != '/' && path.back() != '\\') {
        path = path + "/";
    }

    std::set<std::string> fileNames = {path + "adapter_model.safetensors"};
    SafeTensors safetensors(fileNames);
}