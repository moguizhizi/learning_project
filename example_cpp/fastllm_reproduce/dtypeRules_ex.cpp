#include "file_utils.hpp"
#include "json11.hpp"
#include <string>
#include <vector>

int main() {
    std::string dtypeConfigString;

    std::vector<std::pair<std::string, std::string>> dtypeRules;
    std::string error;

    if (dtypeConfigString.size() > 0) {
        auto dtypeConfig = json11::Json::parse(dtypeConfigString, error);
        if (error != "") {
            printf("Parse dtype config faild.\n");
            printf("config = %s\n", dtypeConfigString.c_str());
            printf("error = %s\n", error.c_str());
        } else {
            for (auto &it : dtypeConfig.array_items()) {
                dtypeRules.push_back(std::make_pair(it['key'].string_value(), it['dtype'].string_value()));
            }
        }
    }

    if (dtypeRules.size() > 0) {
        printf("Dtype rules:\n");
        for (auto &it : dtypeRules) {
            printf("%s: %s\n", it.first.c_str(), it.second.c_str());
        }
    }
}