#include "Timer.h"
#include <iostream>
#include <iomanip> // For std::setw

// Constructor
Timer::Timer(const std::string &name, bool auto_print)
    : m_name(name), m_auto_print(auto_print), m_start_time(Clock::now()), m_running(true)
{
}

// Destructor
Timer::~Timer()
{
    if (m_running && m_auto_print)
    {
        Stop();
        Print();
    }
}

void Timer::Stop()
{
    m_end_time = Clock::now();
    m_running = false;
}

void Timer::Reset()
{
    m_start_time = Clock::now();
    m_running = true;
}

double Timer::ElapsedMillis() const
{
    TimePoint end_point = m_running ? Clock::now() : m_end_time;
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_point - m_start_time);
    return duration.count();
}

void Timer::Print(std::string info) const
{
    std::cout << "[TIMER] " << std::left << std::setw(20) << (info.empty() ? m_name : info)
              << ": " << ElapsedMillis() / 1000.0 << " s" << std::endl;
}

// Helper implementation for the benchmark print
void Timer::PrintBenchmark(const std::string &name, double total_ms, int iterations) const
{
    double avg = total_ms / iterations;
    std::cout << "[BENCH] " << std::left << std::setw(20) << name
              << "| Total: " << total_ms << " ms"
              << "| Avg: " << avg << " ms" << std::endl;
}