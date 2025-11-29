#include <vector>

class Solution {
   public:
    bool canJump(std::vector<int> &nums) {
        int maxStep = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (i + nums[i] > maxStep) {
                maxStep = i + nums[i];
            }

            if (nums[i] == 0 && i != nums.size() - 1 && maxStep <= i) {
                return false;
            }
        }

        return true;
    }
};