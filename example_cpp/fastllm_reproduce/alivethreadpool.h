#pragma once

struct MultiThreadBaseOp {
    virtual void Run() = 0;
};

struct AliveThreadTask {
    int signal;
    MultiThreadBaseOp *op;

    AliveThreadTask();
};

struct AliveThreadPool {
    void PushOp(int tid, MultiThreadBaseOp *op);

    void Wait(int tid);

    void Shutdown();
};