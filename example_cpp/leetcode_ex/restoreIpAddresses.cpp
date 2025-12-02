#include <iostream>
#include <sstream>
#include <vector>

struct insertPos {
    int first;
    int second;
    int third;
};

class Solution {
   public:
    std::vector<insertPos> getTriples(const int len) {
        std::vector<insertPos> posSet;

        insertPos pos;

        for (int i = 1; i < 4; i++) {
            for (int j = i + 1; j < std::min(i + 1 + 3, len); j++) {
                for (int k = j + 1; k < std::min(j + 1 + 3, len); k++) {
                    if (len - 1 - k >= 3) {
                        continue;
                    }

                    pos.first = i;
                    pos.second = j;
                    pos.third = k;

                    posSet.push_back(pos);
                }
            }
        }

        return posSet;
    }

    bool isValidIP(const std::string &ip) {
        std::stringstream ss(ip);
        std::string part;
        std::vector<std::string> parts;

        while (std::getline(ss, part, '.')) {
            if (part.find("0") == 0 && part.length() != 1) {
                return false;
            }

            if (std::stoi(part) > 255) {
                return false;
            }
        }

        return true;
    }

    std::vector<std::string> restoreIpAddresses(std::string s) {
        std::vector<std::string> result;
        const int len = s.length();

        if (len < 4 || len > 12) {
            return result;
        }

        std::vector<insertPos> posSet = getTriples(len);

        for (auto &pos : posSet) {
            int first = pos.first;
            int second = pos.second;
            int third = pos.third;

            int subStr0Len = first - 0;
            int subStr1Len = second - first;
            int subStr2Len = third - second;
            int subStr3Len = len - subStr0Len - subStr1Len - subStr2Len;

            std::string ip = s.substr(0, subStr0Len) + "." + s.substr(first, subStr1Len) + "." + s.substr(second, subStr2Len) + "." +
                             s.substr(third, subStr3Len);
            if (isValidIP(ip)) {
                result.push_back(ip);
            }
        }

        return result;
    }
};