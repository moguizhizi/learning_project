#include <algorithm>
#include <vector>

// class Solution {
//    public:
//     int majorityElement(std::vector<int> &nums) {
//                 std::unordered_map<int, int> repeat_map;
//         for (int i = 0; i < nums.size(); i++) {
//             const int value = nums[i];
//             if (repeat_map.find(value) == repeat_map.end()) {
//                 repeat_map.insert({value, 1});
//             } else {
//                 int nums_repeat = repeat_map[value];
//                 nums_repeat++;
//                 repeat_map[value] = nums_repeat;

//                 if (nums_repeat > (nums.size() / 2)) {
//                     return value;
//                 }
//             }
//         }

//         return nums[0];
//     }
// };

class Solution {
   public:
    int majorityElement(std::vector<int> &nums) {
        const int len = nums.size();
        std::sort(nums.begin(), nums.end());
        return nums[len / 2];
    }
};

// 官方
// class Solution {
//    public:
//     int majorityElement(vector<int> &nums) {
//         int ans = 0, count = 0;
//         for (int x : nums) {
//             if (count == 0) {
//                 ans = x;
//                 count = 1;
//             } else {
//                 count += x == ans ? 1 : -1;
//             }
//         }
//         return ans;
//     }
// };
