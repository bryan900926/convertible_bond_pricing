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
// args : ticker
int main(int argc, char** argv) {

    if (argc < 2) {
        throw std::runtime_error("Usage: " + std::string(argv[0]) + " <ticker> <market_data_file>");
    }

    signal(SIGSEGV, crash_handler); // Memory violations
    signal(SIGFPE, crash_handler);  // Math crashes
    signal(SIGABRT, crash_handler); // Abort calls
    signal(SIGILL, crash_handler);  // Illegal instructions

    #ifdef _WIN32
        unsigned int current_word;
        // _EM_OVERFLOW: The specific bit for overflow
        // _MCW_EM: The mask for all exception control bits
        _controlfp_s(&current_word, _MCW_EM, _MCW_EM);
    #endif

    PricingEngine engine = PricingEngine::Create("D:\\Users\\YYLee\\cb_cpp\\res_table.csv");

    std::vector<std::string> tickers = {
      "EXPE",
      "FSLY",
      "VERI",
      "SPOT",
      "DHR",
      "COLL",
      "EEFT",
      "LAB"
    };
    std::unordered_map<std::string, std::pair<double, double>> dp_bounds = {
        {"COLL", {0.001, 0.01}}, {"DHR", {0.001, 0.01}},
        {"EEFT", {0.001, 0.01}}, {"EXPE", {0.001, 0.01}},
        {"FSLY", {0.01, 0.2}},   {"SPOT", {0.05, 0.15}},
        {"LAB", {0.001, 0.01}},  {"VERI", {0.001, 0.01}}};

    std::unordered_map<std::string, std::pair<double, double>> nu_bounds = {
        {"COLL", {0.0583458, 0.0543022}},
        {"DHR",  {0.0951958, 0.010315}},
        {"EEFT", {0.0693703, 0.061359}},
        {"EXPE", {0.0542259, 0.0463295}},
        {"FSLY", {0.318432,  0.24065}},
        {"SPOT", {0.0938988, 0.0798607}},
        {"LAB",  {3.35939,   3.33788}},
        {"VERI", {0.232296,  0.204105}}
    };

    // std::vector<double> nus;


    // for (int i = 0; i < 5; ++i) {
    //     nus.push_back(nu_bounds["EXPE"].first + (nu_bounds["EXPE"].second - nu_bounds["EXPE"].first) * i / 4.0);
    // }

    // auto config = ConfigFactory::get_EXPE();

    // config.cb.dt_other = 1 / 52.0;
    // config.cdg.v = nus[1];
    // engine.run_pricing_suite("EXPE", config.cb, config.cdg, config.vas);
    // for (const auto &ticker : tickers) {
    //     auto value = nu_bounds[ticker];
    //     value.second, value.first);
    // }

    std::string target_ticker = argv[1];

    if (nu_bounds.find(target_ticker) == nu_bounds.end())
    {
        throw std::runtime_error("Error: Ticker " + target_ticker + " not found in nu_bounds!");
    }
    engine.sensitivity_analysis(target_ticker,
                                "sensitivity_" + target_ticker + ".csv",
                                nu_bounds.at(target_ticker).second,
                                nu_bounds.at(target_ticker).first);

    return 0;
}