#include <numa.h>

int main() {
    struct bitmask *mask = numa_bitmask_alloc(numa_num_configured_nodes());
    numa_bitmask_clearall(mask);
    numa_bitmask_isbitset(mask, 0);
    numa_bind(mask);
    numa_bitmask_free(mask);

    return 0;
}