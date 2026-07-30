#include <Eigen/Dense>
#include <csignal>
#include <iostream>
#include <float.h>
#include <rapidcsv.h>
#include "src/Util/ParameterLoader.hpp"
#include "src/Pricing/PricingEngine.hpp"

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

// args : ticker
int main(int argc, char **argv)
{
    signal(SIGSEGV, crash_handler); // Memory violations
    signal(SIGFPE, crash_handler);  // Math crashes
    signal(SIGABRT, crash_handler); // Abort calls
    signal(SIGILL, crash_handler);  // Illegal instructions

    ParameterLoader<FinalResultMemoSave> loader = ParameterLoader<FinalResultMemoSave>::StartLoadParams(
        "LAB", "D:\\Users\\YYLee\\cb_cpp\\para_test.csv", "D:\\Users\\YYLee\\cb_cpp\\call_para.csv");
    loader.Save("D:\\Users\\YYLee\\cb_cpp\\result.csv", ParameterLoader<FinalResultMemoSave>::FileMode::Overwrite);
    return 0;
}
void test_eigen_array_addition() {
    Eigen::ArrayXXd a(10000000, 20);
    Eigen::ArrayXXd b(10000000, 20);
    for (int i = 0; i < 20; ++i) {
      a += b;
    }
}
void test_eigen_array_addtion_openmp() {
    Eigen::ArrayXXd a(10000000, 20);
    Eigen::ArrayXXd b(10000000, 20);
    #pragma omp parallel for
    for (int i = 0; i < 20; ++i) {
      a += b;
    }
}

