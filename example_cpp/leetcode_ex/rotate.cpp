#include <vector>

class Solution {
   public:
    void rotate(std::vector<int> &nums, int k) {
        if (k == 0) {
            return;
        }

        const int len = nums.size();

        std::vector<int> index(len, 0);
        std::vector<int> cp_nums(len, 0);

        int map_index = 0;
        for (int i = 0; i < len; i++) {
            map_index = (i + k) % len;
            index[map_index] = i;
            cp_nums[i] = nums[i];
        }

        for (int i = 0; i < len; i++) {
            nums[i] = cp_nums[index[i]];
        }
    }
};