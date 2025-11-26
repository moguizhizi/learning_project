
#include <vector>

class Solution {
   public:
    void merge(std::vector<int> &nums1, int m, std::vector<int> &nums2, int n) {
        if (n == 0) {
            return;
        }

        if (m == 0 && n != 0) {
            for (int i = 0; i < n; ++i) {
                nums1[i] = nums2[i];
            }

            return;
        }

        for (int i = m - 1; i >= 0; --i) {
            nums1[n + i] = nums1[i];
        }

        int curPoint = 0;
        int nums1Point = n;
        int nums2Point = 0;

        while (curPoint < (n + m)) {
            if (nums1Point >= (n + m) || nums2Point >= n) {
                break;
            }

            if (nums1[nums1Point] <= nums2[nums2Point]) {
                nums1[curPoint] = nums1[nums1Point];
                curPoint++;
                nums1Point++;
            } else {
                nums1[curPoint] = nums2[nums2Point];
                curPoint++;
                nums2Point++;
            }
        }

        while (nums2Point < n) {
            nums1[curPoint] = nums2[nums2Point];
            curPoint++;
            nums2Point++;
        }
    }
};