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
        int headpoint = 0;
        while (curpoint < len) {
            if ((nums[headpoint] == nums[curpoint]) && ((savepoint - headpoint) < 2)) {
                nums[savepoint] = nums[curpoint];
                savepoint++;
            }

            if (nums[headpoint] != nums[curpoint]) {
                nums[savepoint] = nums[curpoint];
                headpoint = savepoint;
                savepoint++;
            }

            curpoint++;
        }

        return savepoint;
    }
};

// 官方
// class Solution {
//    public:
//     int removeDuplicates(vector<int> &nums) {
//         if (nums.size() == 1) return 1;
//         int count = 2;
//         for (int i = 2; i < nums.size(); i++) {
//             if (nums[i] != nums[count - 2]) {
//                 // count++;
//                 nums[count++] = nums[i];
//             }
//         }
//         return count;
//     }
// };