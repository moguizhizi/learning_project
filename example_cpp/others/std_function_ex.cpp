#include <functional>
#include <iostream>

int func(int a, int b) { return a + b; }

struct AddFuc {
    int operator()(int a, int b) { return a + b; }
};

int main() {
    std::function<int(int, int)> f;
    f = func;

    std::cout << f(1, 2) << std::endl;

    f = [](int a, int b) { return a + b; };
    std::cout << f(1, 2) << std::endl;

    f = AddFuc();
    std::cout << f(1, 2) << std::endl;
}