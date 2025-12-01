
#include <vector>

// O(n^2)
// class Solution {
//    public:
//     int jump(std::vector<int> &nums) {
//         std::map<int, int> posStep;
//         const int len = nums.size();

//         for (int i = 0; i < len; i++) {
//             int step = nums[i];
//             if (step == 0) {
//                 continue;
//             }

//             if (posStep.find(i) == posStep.end()) {
//                 posStep[i] = 0;
//             }

//             if (i == len - 1) {
//                 break;
//             }

//             int minStep = posStep[i];
//             int start = i + 1;
//             int end = i + step;
//             if (end >= len - 1) {
//                 posStep[len - 1] = minStep + 1;
//                 break;
//             }

//             for (int j = start; j <= end; j++) {
//                 if (posStep.find(j) == posStep.end()) {
//                     posStep[j] = posStep[i] + 1;
//                 } else {
//                     posStep[j] = posStep[j] > posStep[i] + 1 ? posStep[i] + 1 : posStep[j];
//                 }
//             }
//         }
//         return posStep[len - 1];
//     }
// };

// O(n)
class Solution {
   public:
    int jump(std::vector<int> &nums) {
        const int len = nums.size();
        int maxPostion = 0;
        int end = 0;
        int step = 0;
        for (int i = 0; i < len - 1; i++) {
            maxPostion = maxPostion < i + nums[i] ? i + nums[i] : maxPostion;
            if (i == end) {
                end = maxPostion;
                ++step;
            }
        }
        return step;
    }
};