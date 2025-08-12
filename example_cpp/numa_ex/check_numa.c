#include <numa.h>
#include <stdio.h>

int main() {
    struct bitmask *valid_nodes = numa_get_mems_allowed();
    int node = 10;
    if (numa_bitmask_isbitset(valid_nodes, node)) {
        printf("Node %d 允许分配内存\n", node);
    } else {
        printf("Node %d 不允许分配内存\n", node);
    }

    return 0;
}