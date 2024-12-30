#ifndef PROFILER_H
#define PROFILER_H

#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <mutex>
#include <functional>

class Profiler {
public:
    // Singleton instance retrieval
    static Profiler& getInstance();

    // Delete copy constructor and assignment operator for singleton
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    // Start and stop profiling
    void start(const std::string& taskName);
    void stop(const std::string& taskName);

    // Log CUDA memory usage
    void logMemoryUsage(const std::string& label = "");

    // Print summary at the end
    void printSummary() const;

    // Generalized profiling for any callable (lambda or function)
    void profile(const std::string& taskName, std::function<void()> func);

private:
    Profiler();            // Private constructor
    ~Profiler();           // Destructor prints profiling summary

    // Helper for tracking tasks
    struct TaskInfo {
        std::string name;
        double duration; // in microseconds
        double preciseDuration; // in nanoseconds
    };

    std::chrono::high_resolution_clock::time_point startTime;
    std::string currentTask;

    std::vector<TaskInfo> taskRecords;

    // Helper for memory profiling
    static void checkCudaMemory(size_t& freeMem, size_t& totalMem);

    // For thread safety (if necessary)
    static std::mutex mutex_;
};

#endif // PROFILER_H
