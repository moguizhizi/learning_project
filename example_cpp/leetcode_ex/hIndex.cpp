#include <algorithm>
#include <vector>

class Solution {
   public:
    int hIndex(std::vector<int> &citations) {
        std::sort(citations.begin(), citations.end(), std::greater<int>());
        int num = 1;
        for (auto &cite : citations) {
            if (cite >= num) {
                num++;
            } else {
                break;
            }
        }

        return num - 1;
    }
};