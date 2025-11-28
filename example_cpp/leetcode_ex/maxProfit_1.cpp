
#include <climits>
#include <vector>

class Solution {
   public:
    int maxProfit(std::vector<int> &prices) {
        int maxProfit = 0;
        int minValue = INT_MAX;
        for (auto &it : prices) {
            if (minValue > it) {
                minValue = it;
            }

            if ((it - minValue) > maxProfit) {
                maxProfit = it - minValue;
            }
        }

        return maxProfit;
    }
};