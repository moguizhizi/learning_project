#include <numa.h>
#include <stdio.h>

int main() {
    if (numa_available() == -1) {
        printf("NUMA not supported\n");
        return 1;
    }

    // 创建一个 bitmask，大小为最大节点数
    struct bitmask *bm = numa_allocate_nodemask();

    // 只允许使用 NUMA 节点 0
    numa_bitmask_clearall(bm);
    numa_bitmask_setbit(bm, 0);

    // 设置内存绑定
    numa_set_membind(bm);

    printf("Memory will now be allocated only from node 0\n");

    // 分配一些内存（将从节点 0 分配）
    void *p = malloc(1024 * 1024);

    numa_free_nodemask(bm);
    free(p);
    return 0;
}
