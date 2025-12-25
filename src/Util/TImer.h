#pragma once

#include <chrono>
#include <string>
#include <functional> // Needed for std::function

class Timer
{
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    explicit Timer(const std::string &name = "Timer", bool auto_print = true);
    ~Timer();

    void Stop();
    void Reset();
    double ElapsedMillis() const;
    void Print(std::string info = "") const;

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
    void PrintBenchmark(const std::string &name, double total_ms, int iterations) const;

    std::string m_name;
    bool m_auto_print;
    TimePoint m_start_time;
    TimePoint m_end_time;
    bool m_running;
};