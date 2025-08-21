#include "alivethreadpool.h"

AliveThreadTask::AliveThreadTask() {
    signal = 0;
    op = nullptr;
}

void AliveThreadPool::PushOp(int tid, MultiThreadBaseOp *op) {}

void AliveThreadPool::Wait(int tid) {}

void AliveThreadPool::Shutdown() {}
