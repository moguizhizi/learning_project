#include <map>
#include <vector>

class Solution {
   public:
    int jump(std::vector<int> &nums) {
        std::map<int, int> posStep;
        const int len = nums.size();

        for (int i = 0; i < len; i++) {
            if (nums[i] == 0) {
                continue;
            }

            if (posStep.find(i) == posStep.end()) {
                posStep[i] = 0;
            }

            int minStep = posStep[i];

            if (i + nums[i] >= len - 1) {
                return minStep + 1;
            }

            for (int j = 1; j <= nums[i]; j++) {
                if (posStep.find(j) == posStep.end()) {
                    posStep[j] = posStep[i] + 1;
                } else {
                    posStep[j] = posStep[j] < posStep[i] + 1 ? posStep[j] : posStep[i] + 1;
                }
            }
        }

        return 0;
    }
};