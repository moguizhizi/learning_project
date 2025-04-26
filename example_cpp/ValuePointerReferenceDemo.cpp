#include <iostream>

void by_value(int x) { x = 20; }
void by_pointer(int* ptr) { *ptr = 30; }
void by_reference(int& ref) { ref = 40; }

int main() {
  int x = 10;

  // 按值传递
  by_value(x);
  std::cout << "After by_value: " << x << std::endl;  // 输出：10

  // 按指针传递
  by_pointer(&x);
  std::cout << "After by_pointer: " << x << std::endl;  // 输出：30

  // 按引用传递
  by_reference(x);
  std::cout << "After by_reference: " << x << std::endl;  // 输出：40

  // 指针和引用的区别
  int y = 50;
  int* ptr = &x;
  ptr = &y;  // 指针可以改变指向
  std::cout << "Pointer points to y: " << *ptr << std::endl;  // 输出：50

  int& ref = x;
//   ref = y;  // 不会改变绑定，只会将 y 的值赋给 x
  std::cout << "Reference still binds to x: " << ref << " " << x << std::endl;  // 输出：50 50

  return 0;
} 