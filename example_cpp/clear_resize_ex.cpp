#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {1, 2, 3};
    
    vec.clear();  // size=0, capacity可能保留
    std::cout << "After clear(): size=" << vec.size() 
              << ", capacity=" << vec.capacity() << std::endl;

    vec = {4, 5, 6};
    vec.resize(0); // 效果同clear()
    std::cout << "After resize(0): size=" << vec.size() 
              << ", capacity=" << vec.capacity() << std::endl;

    vec.shrink_to_fit(); // 释放内存
    std::cout << "After shrink_to_fit(): capacity=" << vec.capacity() << std::endl;
}