#include <numa.h>
int main() {
    numa_run_on_node(0); // 只让进程在节点 0 的 CPU 上跑
    return 0;
}