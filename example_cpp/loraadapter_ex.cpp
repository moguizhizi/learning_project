#include "file_utils.hpp"
#include "struct_space.hpp"

int main() {
    std::string lorapath = "/home/temp";
    std::string path = lorapath;

    if (path.back() != '/' && path.back() != '\\') {
        path = path + "/";
    }

    std::set<std::string> fileNames = {path + "adapter_model.safetensors"};
    SafeTensors safetensors(fileNames);

    std::map<std::string, std::pair<std::string, std::string>> loraDicts;
    float loraScaling;
    for (auto &it : safetensors.GetSortedItemNames()) {
        if (it.size() > 31) {
            if (it.substr(0, 17) == "base_model.model." &&
                (it.substr(it.size() - 14) == ".lora_A.weight" || it.substr(it.size() - 14) == ".lora_B.weight")) {
                std::string originalName = it.substr(17, it.size() - 31);
                if (it.substr(it.size() - 14) == ".lora_A.weight") {
                    loraDicts[originalName].first = it;
                } else {
                    loraDicts[originalName].second = it;
                }
            }
        }
    }

    std::string loraError;
    auto loraConfig = json11::Json::parse(ReadAllFile(path + "adapter_config.json"), loraError);
    loraScaling = loraConfig["lora_alpha"].number_value() / loraConfig["r"].number_value();
    std::cout << "loraScaling:" << loraScaling << std::endl;
}