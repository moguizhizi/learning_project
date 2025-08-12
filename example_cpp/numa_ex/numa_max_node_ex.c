#include <numa.h>
#include <stdio.h>

int main() {
    printf("总的节点数 %d\n", numa_max_node());
    return 0;
}