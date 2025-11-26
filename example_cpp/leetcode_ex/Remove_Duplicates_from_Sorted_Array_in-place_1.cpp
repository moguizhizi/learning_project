#include <vector>

class Solution {
   public:
    int removeDuplicates(std::vector<int> &nums) {
        const int len = nums.size();
        if (len == 0) {
            return 0;
        }

        int curpoint = 0;
        int savepoint = 0;
        int value;

        while (curpoint < len) {
            value = nums[savepoint];
            if (nums[curpoint] != value) {
                savepoint++;
                if (savepoint != curpoint) {
                    nums[savepoint] = nums[curpoint];
                }
            }
            curpoint++;
        }

        return savepoint + 1;
    }
};