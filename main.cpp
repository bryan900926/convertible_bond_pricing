#include <Eigen/Dense>
#include <csignal>
#include <iostream>
#include <float.h>
#include <rapidcsv.h>
#include "src/Util/ParameterLoader.hpp"

void crash_handler(int signal_number)
{
    std::cerr << "\n[CRASH LOG] Program died from OS Signal: " << signal_number << "\n";
    if (signal_number == SIGSEGV)
        std::cerr << "Reason: Segmentation Fault (Bad Memory Access)\n";
    if (signal_number == SIGFPE)
        std::cerr << "Reason: Arithmetic Exception (Math Error)\n";
    if (signal_number == SIGILL)
        std::cerr << "Reason: Illegal Instruction (Bad CPU architecture flag)\n";
    std::exit(signal_number);
}

int main(int argc, char **argv)
{
    signal(SIGSEGV, crash_handler); // Memory violations
    signal(SIGFPE, crash_handler);  // Math crashes
    signal(SIGABRT, crash_handler); // Abort calls
    signal(SIGILL, crash_handler);  // Illegal instructions

    ParameterLoader<FinalResultMemoSave> loader = ParameterLoader<FinalResultMemoSave>::StartLoadParams(
        "LAB", "D:\\Users\\YYLee\\cb_cpp\\para_test.csv", "D:\\Users\\YYLee\\cb_cpp\\schedule_para.csv");
    loader.Pricing("D:\\Users\\YYLee\\cb_cpp\\result.csv", ParameterLoader<FinalResultMemoSave>::FileMode::Overwrite);
    return 0;
}

