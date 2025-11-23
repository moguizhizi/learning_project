#include "alivethreadpool.h"

#include "struct_space.hpp"

AliveThreadTask::AliveThreadTask() {
    signal = 0;
    op = nullptr;
}

void AliveThreadPool::PushOp(int tid, MultiThreadBaseOp *op) {}

void AliveThreadPool::Wait(int tid) {}

void AliveThreadPool::Shutdown() {}

void RunMultiThreadMemcpyMultiLines(std::vector<MultiThreadMemcpyMultiLinesTask> &tasks, AliveThreadPool *pool) {
    int threadNum = pool->threads.size();
    int n = tasks.size();
    int per = n / pool->threads.size();
    int cur = 0;
    std::vector<MultiThreadMemcpyMultiLinesOp *> ops;
    for (int i = 0; i < threadNum; i++) {
        int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
        ops.push_back(new MultiThreadMemcpyMultiLinesOp(tasks.data(), cur, end));
        cur = end;
    }
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(i, ops[i]);
    }
    for (int i = 0; i < threadNum; i++) {
        pool->Wait(i);
        delete ops[i];
    }
}
