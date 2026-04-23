#define DEBUG
#include <Eigen/Dense>
#include "src/HullWhiteModel/HullWhiteModel.h"
#include "src/Pricing/CbModel.h"
#include <fstream>
#include "src/Util/Util.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <string>

std::unordered_map<std::string, std::unordered_map<std::string, double>> market_data;

#ifdef DEBUG
const std::vector<double> dts = { 1.0 / 12 };
#else
const std::vector<double> dts = { 1, 1.0 / 2, 1.0 / 3, 1.0 / 12, 1.0 / 24, 1.0 / 36, 1.0 / 48 };
#endif

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

// Trim whitespace and Windows \r from strings
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, (last - first + 1));
}

// Safely fetch market data, returning 0.0 if missing to prevent out_of_range crashes
double get_market_val(const std::string& ticker, const std::string& key) {
    auto it_ticker = market_data.find(ticker);
    if (it_ticker != market_data.end()) {
        auto it_key = it_ticker->second.find(key);
        if (it_key != it_ticker->second.end()) {
            return it_key->second;
        }
    }
    std::cerr << "[Warning] Missing data for " << ticker << " key: " << key << "\n";
    return 0.0; 
}

// Load market data from CSV
bool load_market_data(const std::string& filepath) {
    std::ifstream infile(filepath);
    if (infile.fail()) {
        std::cerr << "Error opening file: " << filepath << "\n";
        return false;
    }

    std::string line;
    std::vector<std::string> col_names;

    // 1. Parse Headers
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string name;
        while (std::getline(iss, name, ',')) {
            col_names.push_back(trim(name));
        }
    }
    if (!col_names.empty()) col_names.pop_back(); // Remove Ticker column header

    // 2. Parse Rows
    while (std::getline(infile, line)) {
        if (trim(line).empty()) continue;

        std::istringstream iss(line);
        std::string segment;
        std::vector<std::string> row_values;

        while (std::getline(iss, segment, ',')) {
            row_values.push_back(trim(segment));
        }
        if (row_values.empty()) continue;

        std::string ticker = row_values.back();
        row_values.pop_back();

        std::unordered_map<std::string, double> data_map;
        for (size_t i = 0; i < row_values.size() && i < col_names.size(); ++i) {
            try {
                data_map[col_names[i]] = std::stod(row_values[i]);
            } catch (...) {
                data_map[col_names[i]] = 0.0;
            }
        }
        market_data[ticker] = data_map;
    }
    return true;
}

// ==========================================
// CORE PRICING ENGINE
// ==========================================

void run_pricing_suite(const std::string& ticker, CbParas cb, CdgParas cdg, VasciekParas vas) {
    // 1. Inject Dynamic Market Data
    cb.sigma_V  = get_market_val(ticker, "sigma_V");
    cdg.lamda   = get_market_val(ticker, "lambda");
    cdg.phi     = get_market_val(ticker, "phi");
    cdg.V0      = get_market_val(ticker, "V0");
    cdg.sigma_v = get_market_val(ticker, "sigma_V");
    cb.rho = get_market_val(ticker, "rho");
    cdg.v = get_market_val(ticker, "v");
    std::printf("[Info] %s - Injected Market Data: sigma_V=%.4f, lambda=%.4f, phi=%.4f, V0=%.4f, rho=%.4f, v=%.4f\n", 
                ticker.c_str(), cb.sigma_V, cdg.lamda, cdg.phi, cdg.V0, cb.rho, cdg.v);
    // 2. Setup Output File
    std::string out_file = ticker + ".csv";
    std::ofstream data(out_file);
    if (!data.is_open()) {
        std::cerr << "Failed to open output file: " << out_file << "\n";
        return;
    }
    
    // Add market baseline headers if needed, otherwise standard headers
    data << "dt,cb_price,e_forward_price,e_backward_price,default_prob\n";

    // 3. Execute Pricing Loop
    std::cout << "Pricing " << ticker << "...\n";
    for (double dt : dts) {
        cb.dt_other = dt;
        try {
            FinalResult res = CbTreePricing(cb, cdg, vas);
            data << dt << "," << res.cb_price << "," 
                 << res.equity_forward_price << "," 
                 << res.equity_backward_price << "," 
                 << res.default_prob << "\n";
        } catch (const std::exception& e) {
            std::cerr << "  -> Error pricing " << ticker << " with dt=" << dt << ": " << e.what() << "\n";
        }
    }
    std::cout << "Finished " << ticker << ". Results saved to " << out_file << "\n\n";
}

// ==========================================
// MAIN EXECUTION
// ==========================================

int main() {
    if (!load_market_data("D:\\Users\\YYLee\\cb_cpp\\res_table.txt")) {
        return 1;
    }
    auto price_LAB = []() {
        std::vector<CallInfo> calls = {CallInfo("2016-02-01", 130.0), CallInfo("2021-02-05", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2016-02-01");
        CbParas cb = { 18, 0, 100, 0.4954, 1.78750, 28844000, 2013000, 100, 20, 0, 20, false, 0.0275, 1.0/12, 1, call_schedule };
        CdgParas cdg = { 0.7, 0, std::log(0.45101481855527364), 0, 0.010226007, 0, 1.81 };
        VasciekParas vas = { 0.136179596167386, 0.0447875855396496, 0.0200316752383784, 0.00393433054895577 };
        run_pricing_suite("LAB", cb, cdg, vas);
    };

    auto price_EEFT = []() {
        std::vector<CallInfo> calls = {CallInfo("2022-09-20", 130.0), CallInfo("2025-03-20", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2021-03-15");
        CbParas cb = { 28, 0, 100, 0.4456, 0.52987, 52753000, 4522280, INT_MAX, 20, 0, 20, false, 0.0075, 1.0/12, 1, call_schedule};
        CdgParas cdg = { 0.7, 0, std::log(0.28741741636296036), 0, 0.003567945, 0, 1.81 };
        VasciekParas vas = { 0.22336291527447, 0.0339962229038466, 0.0381091909084148, 0.000879559544106292 };
        run_pricing_suite("EEFT", cb, cdg, vas);
    };

    auto price_DHR = []() {
      std::vector<CallInfo> calls = {
          CallInfo("2010-01-22", 77.128), CallInfo("2011-01-22", 78.970),
          CallInfo("2012-01-22", 80.857), CallInfo("2013-01-22", 82.789),
          CallInfo("2014-01-22", 84.767), CallInfo("2015-01-22", 86.792),
          CallInfo("2016-01-22", 88.865), CallInfo("2017-01-22", 90.989),
          CallInfo("2018-01-22", 93.162), CallInfo("2019-01-22", 95.388),
          CallInfo("2020-01-22", 97.667)};
        calls = {CallInfo("2010-01-22", 10000000)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2009-01-22");
        CbParas cb = {
          12.0,
          0,
          100,
          0.4954,
          1.45352,
          354487000,
          6200000,
          771,
          20,
          0,
          20,
          false,
          0.0,
          1.0/12,
          1,
          call_schedule
        };
        CdgParas cdg = { 0.7, 0, std::log(0.39883480752180996), 0, 0.004711, 0, 1.81 };
        VasciekParas vas = { 0.386248236006081, 0.0469941534383253, 0.0593403428043975, 0.00260058521741104 };
        run_pricing_suite("DHR", cb, cdg, vas);
    };

    auto price_COLL = []() {
        std::vector<CallInfo> calls = {CallInfo("2023-02-15", 130.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2021-02-16");
        CbParas cb = { 7.0, 0, 100, 0.4456, 3.67815, 34756000, 1437500, 130, 20, 0, 20, false, 0.02625, 1.0/12, 1, call_schedule};
        CdgParas cdg = { 1.7, 0, std::log(0.357547605), 0, 0.025091036, 0, 1 };
        VasciekParas vas = { 0.01628559909171213, 0.1595220619446585, 0.00428873644512, 0.0009307909267961937 };
        run_pricing_suite("COLL", cb, cdg, vas);
    };
    // market cb :109.761 stock price : 211.92999
    auto price_EXPE = []() {
        std::vector<CallInfo> calls = {CallInfo("2024-02-20", 130.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-02-15");
        CbParas cb = { 4.0, 0, 100, 0.4456, 0.39212, 150231000, 986000, 1300, 20, 0, 20, false, 0.0, 1.0/12, 1, call_schedule};
        CdgParas cdg = { 1.7, 0, std::log(0.9060275546106771), 0, 0.009263, 0, 1 };
        VasciekParas vas = { 0.04205976155550541, 0.283762364578435, 0.0221521133952676, 0.00427564701108187 };
        run_pricing_suite("EXPE", cb, cdg, vas);
    };
    auto price_SPOT = []() {
        std::vector<CallInfo> calls = {CallInfo("2024-03-20", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-02-15");
        CbParas cb = { 4.0, 0, 100, 0.4456, 0.1941, 192152000, 1500000, 130, 20, 0, 20, false, 0.0, 1.0/12, 1, call_schedule };
        CdgParas cdg = { 4.9, 0, std::log(0.15723057959785527), 0, 0.001669371, 0, 1.81 };
        VasciekParas vas = { 0.136179596167386, 0.0447875855396496, 0.0831316482986798, 0.00393433054895577 };
        run_pricing_suite("SPOT", cb, cdg, vas);
    };

    auto price_FSLY = []() {
              std::vector<CallInfo> calls = {CallInfo("2024-03-20", 130.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-03-15");
        CbParas cb = { 4.0, 0, 100, 0.4456, 0.97272, 117500000, 9487500, 130, 20, 0, 20, false, 0.0, 1.0/12, 1, call_schedule };
        CdgParas cdg = { 0.7, 0, std::log(0.3554711592587863), 0, 0.001910412, 0, 1.81 };
        VasciekParas vas = { 0.168347138448497, 0.123838434970359, 0.576508365942268, 0.00391899840867379 };
        vas = { 2.73186466686729, 0.0557494010613996, 0.120590482306319, 0.0326238320383245 };
        run_pricing_suite("FSLY", cb, cdg, vas);
        DefaultTest(cb, cdg, vas, 10000);
    };
    // market cb :56.758 stock price : 7.27
    auto price_VERI = []() {
        std::vector<CallInfo> calls = {CallInfo("2024-11-20", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-11-15");
        CbParas cb = { 4.0, 0, 100, 0.4456, 2.72068, 36290000, 2013000, 130, 20, 0, 20, false, 0.0175, 1.0/12, 1, call_schedule };
        CdgParas cdg = { 2.1, 0, std::log(0.6611612151495524), 0, 0.006972534, 0, 0.21 };
        VasciekParas vas = { 2.73186466686729, 0.0557494010613996, 0.120590482306319, 0.0326238320383245 };
        run_pricing_suite("VERI", cb, cdg, vas);
    };

    auto test = []() {
        std::vector<CallInfo> calls = {CallInfo("2024-11-20", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-11-15");
        CbParas cb = { 4.0, 0.2725, 100, 0.4456, 3.67815, 34756000, 1437500, 130, 20, 0.0254, 20, false, 0.02625, 1.0/52, 1, call_schedule };
        CdgParas cdg = {  0.3347, 9.41360, -0.9046, 1440140802, 0.025091036, 0.27250, 2.05071};
        VasciekParas vas = { 0.3562, 0.0208, 0.0520, 0.0021 };
        CbTreePricing(cb, cdg, vas);
        DefaultTest(cb, cdg, vas, 10000);
    };
    // test();
    // price_LAB();
    // price_EEFT();
    // price_DHR();
    // price_COLL();
    // price_EXPE();
    // price_SPOT();
    price_FSLY();
    // price_VERI();
    
    return 0;
}