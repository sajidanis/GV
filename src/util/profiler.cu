#include "profiler.cuh"

#include <iostream>
#include <iomanip>
#include <sstream>

std::mutex Profiler::mutex_;

// Constructor and Destructor
Profiler::Profiler() {}
Profiler::~Profiler() {
    printSummary(); // Print summary at the end of the program
}

Profiler& Profiler::getInstance() {
    static Profiler instance; // Singleton instance
    return instance;
}

void Profiler::start(const std::string& taskName) {
    if (!currentTask.empty()) {
        throw std::runtime_error("A task is already being profiled. Stop it before starting a new one.");
    }
    currentTask = taskName;
    startTime = std::chrono::high_resolution_clock::now();
}

void Profiler::stop(const std::string& taskName) {
    if (currentTask != taskName) {
        throw std::runtime_error("Mismatched task names in start and stop calls.");
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
    TaskInfo taskInfo;
    taskInfo.name = taskName;
    taskInfo.duration = duration.count() / 1000.0;
    taskInfo.preciseDuration = duration.count() / 1.0;
    taskRecords.push_back(taskInfo); // Convert to microseconds
    currentTask.clear();
}

void Profiler::logMemoryUsage(const std::string& label) {
    size_t freeMem, totalMem;
    checkCudaMemory(freeMem, totalMem);

    double freeMB = freeMem / (1024.0 * 1024.0);
    double totalMB = totalMem / (1024.0 * 1024.0);
    double usedMB = totalMB - freeMB;

    std::ostringstream oss;
    oss << "Memory Usage [" << label << "]: ";
    oss << "Total: " << std::fixed << std::setprecision(2) << totalMB << " MB, ";
    oss << "Used: " << usedMB << " MB, ";
    oss << "Free: " << freeMB << " MB";

    taskRecords.push_back({oss.str(), 0}); // Log as a task with no duration
}

void Profiler::printSummary() const {
    std::cout << "\n========== Profiler Summary ==========" << std::endl;
    for (const auto& record : taskRecords) {
        if (record.duration > 0) {
            std::cout << "Task: " << record.name 
                      << " | Duration: " << std::fixed << std::setprecision(3)
                      << record.duration / 1000.0 << " ms" << "\t"
                      << record.duration << " us" << "\t" << record.preciseDuration << " ns" << std::endl;
        } else {
            std::cout << record.name << std::endl; // Memory usage logs
        }
    }
    std::cout << "=======================================" << std::endl;
}

void Profiler::checkCudaMemory(size_t& freeMem, size_t& totalMem) {
    cudaMemGetInfo(&freeMem, &totalMem);
}

void Profiler::profile(const std::string& taskName, std::function<void()> func) {
    start(taskName);
    func(); // Execute the callable (e.g., a lambda)
    cudaDeviceSynchronize(); // Ensure GPU tasks are complete
    stop(taskName);
}
