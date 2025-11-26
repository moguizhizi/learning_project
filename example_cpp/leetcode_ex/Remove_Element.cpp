#include <vector>

class Solution {
   public:
    int removeElement(std::vector<int> &nums, int val) {
        const int len = nums.size();
        if (len == 0) {
            return 0;
        }

        int savepoint = 0;
        int curpoint = 0;
        while (curpoint < len) {
            if (nums[curpoint] != val) {
                if (savepoint != curpoint) {
                    nums[savepoint] = nums[curpoint];
                }
                savepoint++;
            }
            curpoint++;
        }
        return savepoint;
    }
};