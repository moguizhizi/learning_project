
#include <climits>
#include <vector>

class Solution {
   public:
    int maxProfit(std::vector<int> &prices) {
        int minValue = INT_MAX;
        int maxProfit = 0;
        int sum = 0;
        for (auto &it : prices) {
            if (minValue > it) {
                minValue = it;
            }

            if ((it - minValue) > maxProfit) {
                maxProfit = it - minValue;
            } else {
                sum += maxProfit;
                maxProfit = 0;
                minValue = it;
            }
        }

        sum += maxProfit;

        return sum;
    }
};