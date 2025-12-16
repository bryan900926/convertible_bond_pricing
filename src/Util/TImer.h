#pragma once

#include <chrono>
#include <string>
#include <functional> // Needed for std::function

class Timer
{
public:
    // Use high_resolution_clock for best precision
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    // Constructor & Destructor
    explicit Timer(const std::string &name = "Timer", bool auto_print = true);
    ~Timer();

    // Core Functions
    void Stop();
    void Reset();
    double ElapsedMillis() const;
    void Print() const;

    // --- TEMPLATES MUST STAY IN .H ---
    // This function cannot move to .cpp because it uses templates/logic required at compile time.
    static void Benchmark(const std::string &name, std::function<void()> func, int iterations = 1)
    {
        Timer t(name, false);
        for (int i = 0; i < iterations; ++i)
        {
            func();
        }
        t.Stop();
        t.PrintBenchmark(name, t.ElapsedMillis(), iterations);
    }

private:
    // Helper to keep the header clean of iostream (optional optimization)
    void PrintBenchmark(const std::string &name, double total_ms, int iterations) const;

    std::string m_name;
    bool m_auto_print;
    TimePoint m_start_time;
    TimePoint m_end_time;
    bool m_running;
};