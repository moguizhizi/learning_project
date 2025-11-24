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

    // 每个线程平均分配的任务数
    int per = n / threadNum;
    // 剩余任务数，用于负载均衡
    int remainder = n % threadNum;

    int cur = 0;
    std::vector<MultiThreadMemcpyMultiLinesOp *> ops;

    for (int i = 0; i < threadNum; i++) {
        // 计算当前线程需要处理的任务数
        int taskCount = per + (i < remainder ? 1 : 0); // 前 remainder 个线程多处理一个任务
        int end = cur + taskCount;                     // 当前线程的结束位置

        // 创建并分配任务
        ops.push_back(new MultiThreadMemcpyMultiLinesOp(tasks.data(), cur, end));
        cur = end; // 更新起始位置
    }

    // 推送任务到线程池
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(i, ops[i]);
    }

    // 等待线程任务执行完成
    for (int i = 0; i < threadNum; i++) {
        pool->Wait(i);
        delete ops[i];
    }
}

void RunMultiThreadMoeReduce(
    std::vector<std::pair<int, float>> *task, std::vector<float> *tempResult, float *curOutput, int dim, AliveThreadPool *pool) {
    int threadNum = pool->threads.size();
    threadNum = std::min(threadNum, 8); // 限制最大线程数为8

    int n = task->size();
    int per = n / threadNum;       // 每个线程平均分配的任务数
    int remainder = n % threadNum; // 计算剩余任务数，给前面的线程分配额外的任务

    std::vector<MultiThreadMoeReduceOp *> ops;
    int cur = 0;

    // 为每个线程分配任务
    for (int i = 0; i < threadNum; i++) {
        // 计算当前线程需要处理的任务量
        int taskCount = per + (i < remainder ? 1 : 0); // 前 remainder 个线程多处理一个任务
        int end = cur + taskCount;                     // 当前线程的结束位置

        // 创建并分配任务操作
        ops.push_back(new MultiThreadMoeReduceOp(task, tempResult, curOutput, dim, cur, end));
        cur = end; // 更新任务起始位置
    }

    // 将线程任务推送到线程池中
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(i, ops[i]);
    }

    // 等待线程池中所有任务完成
    for (int i = 0; i < threadNum; i++) {
        pool->Wait(i);
        delete ops[i];
    }
}
