#include <numa.h>
#include <stdio.h>

int main() {
    printf("一共有 %d numa 节点", numa_num_configured_nodes());
    return 0;
}