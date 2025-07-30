#include <iostream>
#include <mutex>
#include <thread>

std::mutex cout_mutex;

void Hello(int i) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "Hello world:" << i << std::endl;
}

int main() {
    std::thread t1(Hello, 1);
    std::thread t2(Hello, 2);

    t1.join();
    t2.join();

    std::cout << "All threads finished.\n";
    return 0;
}