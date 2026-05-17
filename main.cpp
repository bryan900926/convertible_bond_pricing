#include <Eigen/Dense>
#include <csignal>
#include <iostream>
#include <float.h>
#include "src/Pricing/PricingEngine.hpp"
#include <string>


void crash_handler(int signal_number) {
    std::cerr << "\n[CRASH LOG] Program died from OS Signal: " << signal_number << "\n";
    if (signal_number == SIGSEGV) std::cerr << "Reason: Segmentation Fault (Bad Memory Access)\n";
    if (signal_number == SIGFPE)  std::cerr << "Reason: Arithmetic Exception (Math Error)\n";
    if (signal_number == SIGILL)
      std::cerr << "Reason: Illegal Instruction (Bad CPU architecture flag)\n";
    std::exit(signal_number);
}

int main() {
    #ifdef _WIN32
        unsigned int current_word;
        // _EM_OVERFLOW: The specific bit for overflow
        // _MCW_EM: The mask for all exception control bits
        _controlfp_s(&current_word, _MCW_EM, _MCW_EM);
    #endif
        
    PricingEngine engine = PricingEngine::Create("./res_table.csv");
    signal(SIGSEGV, crash_handler); // Memory violations
    signal(SIGFPE, crash_handler);  // Math crashes
    signal(SIGABRT, crash_handler); // Abort calls
    signal(SIGILL, crash_handler);  // Illegal instructions

    engine.bisect_dp("EXPE");
    // engine.bisect_dp("SPOT");
    // engine.bisect_dp("FSLY");
    // engine.bisect_dp("VERI");
    // engine.bisect_dp("COLL");
    // auto config = ConfigFactory::get_COLL();
    // config.cb.dt_other = 1.0 / 12.0;
    // engine.run_pricing_suite("COLL", config.cb, config.cdg, config.vas);

    return 0;
}