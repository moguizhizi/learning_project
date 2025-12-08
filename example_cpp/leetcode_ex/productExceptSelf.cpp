#include <vector>

class Solution {
   public:
    std::vector<int> productExceptSelf(std::vector<int> &nums) {
        const int len = nums.size();
        std::vector<int> product(len, 1);

        if (len == 1) {
            return {0};
        }

        int tempPrduct = nums[0];
        for (int i = 1; i < len; ++i) {
            product[i] = tempPrduct;
            tempPrduct *= nums[i];
        }

        tempPrduct = nums[len - 1];
        for (int i = len - 1 - 1; i >= 0; --i) {
            product[i] *= tempPrduct;
            tempPrduct *= nums[i];
        }

        return product;
    }
};