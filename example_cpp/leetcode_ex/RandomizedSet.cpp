#include <cstdlib>
#include <unordered_map>
#include <vector>

class RandomizedSet {
   private:
    std::unordered_map<int, int> mp;
    std::vector<int> v;

   public:
    RandomizedSet() {}

    bool insert(int val) {
        if (mp.count(val)) return false;
        v.push_back(val);
        int n = v.size();
        mp[val] = n - 1;
        return true;
    }

    bool remove(int val) {
        if (!mp.count(val)) return false;
        int p = mp[val];
        int n = v.size();
        if (p == n - 1) {
            mp.erase(val);
            v.pop_back();
            return true;
        }
        v[p] = v[n - 1];
        v.pop_back();
        mp.erase(val);
        mp[v[p]] = p;
        return true;
    }

    int getRandom() {
        int n = v.size();
        int r = rand() % n;
        return v[r];
    }
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet* obj = new RandomizedSet();
 * bool param_1 = obj->insert(val);
 * bool param_2 = obj->remove(val);
 * int param_3 = obj->getRandom();
 */