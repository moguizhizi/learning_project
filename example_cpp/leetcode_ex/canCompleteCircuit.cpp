#include <climits>
#include <vector>

// class Solution {
//    public:
//     int findFirstNonNegativeIndex(int start, int end, std::vector<int> &balance) {
//         int firstNonNegative = end;
//         for (int i = start; i < end; ++i) {
//             if (balance[i] < 0) {
//                 continue;
//             } else {
//                 firstNonNegative = i;
//                 break;
//             }
//         }

//         return firstNonNegative;
//     }

//     int canCompleteCircuit(std::vector<int> &gas, std::vector<int> &cost) {
//         const int len = gas.size();
//         const int balanceLen = len + len - 1;
//         std::vector<int> balance;
//         balance.resize(balanceLen);
//         for (int i = 0; i < balanceLen; ++i) {
//             balance[i] = gas[i % len] - cost[i % len];
//         }

//         int index = -1;
//         int startIndex = findFirstNonNegativeIndex(0, balanceLen, balance);
//         int sum = 0;
//         int cursor = startIndex;
//         while (startIndex < len) {
//             if (cursor - startIndex <= len - 1) {
//                 sum = sum + balance[cursor];
//                 if (sum < 0) {
//                     startIndex = findFirstNonNegativeIndex(cursor + 1, balanceLen, balance);
//                     cursor = startIndex;
//                     sum = 0;
//                 } else {
//                     ++cursor;
//                 }
//             } else {
//                 index = startIndex;
//                 break;
//             }
//         }

//         return index;
//     }
// };

// 官方
// class Solution {
//    public:
//     int canCompleteCircuit(std::vector<int> &gas, std::vector<int> &cost) {
//         int totalGas = 0;  // 总油量
//         int totalCost = 0; // 总消耗
//         int curGas = 0;    // 当前剩余油量
//         int start = 0;     // 候选起点

//         for (int i = 0; i < gas.size(); ++i) {
//             totalGas += gas[i];
//             totalCost += cost[i];
//             curGas += gas[i] - cost[i]; // 累计当前起点的剩余油量

//             // 若当前剩余油量<0，说明从start到i都不能作为起点
//             if (curGas < 0) {
//                 start = i + 1; // 重置起点为i+1
//                 curGas = 0;    // 重置当前剩余油量
//             }
//         }

//         // 总油量不足，直接返回-1
//         return totalGas >= totalCost ? start : -1;
//     }
// };

class Solution {
   public:
    int canCompleteCircuit(std::vector<int> &gas, std::vector<int> &cost) {
        const int len = gas.size();
        int totalGas = 0;
        int totalCost = 0;
        for (int i = 0; i < len; ++i) {
            totalGas += gas[i];
            totalCost += cost[i];
        }

        int index = -1;
        if (totalGas < totalCost) {
            return index;
        }

        int minTotalDiff = INT_MAX;
        int prefixSum = 0;
        for (int i = 0; i < len; ++i) {
            prefixSum += (gas[i] - cost[i]);
            if (minTotalDiff > prefixSum) {
                minTotalDiff = prefixSum;
                index = i + 1;
            }
        }

        return index % len;
    }
};