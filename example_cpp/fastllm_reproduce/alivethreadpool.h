#pragma once

#include <thread>
#include <vector>

struct MultiThreadBaseOp {
    virtual void Run() = 0;
};

struct AliveThreadTask {
    int signal;
    MultiThreadBaseOp *op;

    AliveThreadTask();
};

struct AliveThreadPool {
    std::pair<int, int> curActivateThreadInterval; // 设定当前激活 [curActivateThreadInterval.first, curActivateThreadInterval.second) 的线程
    std::vector<std::thread *> threads;

    void PushOp(int tid, MultiThreadBaseOp *op);

    void Wait(int tid);

    void Shutdown();
};