#include "HullWhiteModel/HullWhiteModel.h"
#include <iostream>
#include "Pricing/CbModel.h"
#include "Util/Util.h"
#include "PricingConfig.hpp"

class PricingEngine {
    const std::vector<double> dts = {1.0 / 52 };
    std::unordered_map<std::string, std::unordered_map<std::string, double>> market_data;
    std::unordered_map<std::string, double> dp_table = {
        {"EXPE", 0.035895},
        {"SPOT", 0.042569},
        {"VERI", 0.040224},
        {"FSLY", 0.042569},
    };
    std::unordered_map<std::string, double> cb_market_table = {
        {"LAB", 105.0},
        {"EEFT", 102.0},
        {"DHR", 84},
        {"COLL", 111.5950},
        {"EXPE", 122.7410},
        {"SPOT", 84.2600},
        {"FSLY", 73.4100},
        {"VERI", 96.0}
    };
    std::unordered_map<std::string, std::function<PricingConfig()>>
        factory_functions = {{"LAB", ConfigFactory::get_LAB},
                             {"EEFT", ConfigFactory::get_EEFT},
                             {"DHR", ConfigFactory::get_DHR},
                             {"COLL", ConfigFactory::get_COLL},
                             {"EXPE", ConfigFactory::get_EXPE},
                             {"SPOT", ConfigFactory::get_SPOT},
                             {"FSLY", ConfigFactory::get_FSLY},
                             {"VERI", ConfigFactory::get_VERI}};
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
public:
    static PricingEngine Create(const std::string& market_data_file) {
        PricingEngine engine;
        if (!engine.load_market_data(market_data_file)) {
            throw std::runtime_error("Failed to load market data from " + market_data_file);
        }
        return engine;
    }
    void sensitivity_analysis(const std::string &ticker,
                              const std::string &file_name, const double lo_nu, const double hi_nu) {
        auto config = factory_functions[ticker]();
        config.cb.sigma_V  = get_market_val(ticker, "sigma_V");
        config.cdg.lamda   =  get_market_val(ticker, "lambda");
        config.cdg.phi     = get_market_val(ticker, "phi");
        config.cdg.V0      = get_market_val(ticker, "V0");
        config.cdg.sigma_v = get_market_val(ticker, "sigma_V");
        config.cb.rho = get_market_val(ticker, "rho");
        config.cdg.v = get_market_val(ticker, "v");
        config.cdg.l0 = std::log(get_market_val(ticker, "debt_ratio_0"));

        std::printf("[Info] %s - Injected Market Data: sigma_V=%.4f, lambda=%.4f, phi=%.4f, V0=%.4f, rho=%.4f, v=%.4f, l0=%.4f\n",
                    ticker.c_str(), config.cb.sigma_V, config.cdg.lamda, config.cdg.phi, config.cdg.V0, config.cb.rho, config.cdg.v, config.cdg.l0);
        // 2. Setup Output File
        std::string out_file = "./result/" + ticker + "_sensitivity.csv";
        std::filesystem::create_directories("./result/");
        std::ofstream data(out_file);
        if (!data.is_open()) {
            std::cerr << "Failed to open output file: " << out_file << "\n";
            return;
        }
        data << "ticker,dt,cb_price,default_prob,dp(monte_carlo),dp(tree),default_periods,lambda,v\n";
        std::vector<double> lambda_mul = {1, 0.1, 0.05};
        std::vector<double> nus;

        for (int i = 0; i < 5; ++i) {
            nus.push_back(lo_nu + (hi_nu - lo_nu) * i / 4.0);
        }

        // 3. Execute Pricing Loop
        std::cout << "Pricing " << ticker << "...\n";
        config.cb.dt_other = 1 / 52.0;
        for (double x : lambda_mul) {
            for (double nu : nus) {
                config.cdg.lamda = get_market_val(ticker, "lambda") * x;
                config.cdg.v = nu;
                double dp1 = DefaultTest(config.cb, config.cdg, config.vas, 10000);
                DefaultTestV1Result dp_result = DefaultTestV1(config.cb, config.cdg, config.vas, 10000);
                try {
                    FinalResultMemoSave res = CbTreePricingMemoSave(config.cb, config.cdg, config.vas, ticker);
                    data << ticker << "," << config.cb.dt_other << "," << res.cb_price
                        << "," << res.default_prob << ","
                        << dp1 << ","
                        << dp_result.average_default_probability << ","
                        << dp_result.average_default_periods << ","
                        << config.cdg.lamda << "," << config.cdg.v
                        << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "  -> Error pricing " << ticker << " with dt=" << config.cb.dt_other << ": " << e.what() << "\n";
                }
            }
        }
        std::cout << "Finished " << ticker << ". Results saved to " << out_file << "\n";
    }
    void run_pricing_suite(const std::string& ticker, CbParas cb, CdgParas cdg, VasciekParas vas) {
        cb.sigma_V  = get_market_val(ticker, "sigma_V");
        cdg.lamda   =  get_market_val(ticker, "lambda");
        cdg.phi     = get_market_val(ticker, "phi");
        cdg.V0      = get_market_val(ticker, "V0");
        cdg.sigma_v = get_market_val(ticker, "sigma_V");
        cb.rho = get_market_val(ticker, "rho");
        cdg.v = get_market_val(ticker, "v");
        cdg.l0 = std::log(get_market_val(ticker, "debt_ratio_0"));

        std::printf("[Info] %s - Injected Market Data: sigma_V=%.4f, lambda=%.4f, phi=%.4f, V0=%.4f, rho=%.4f, v=%.4f, l0=%.4f\n",
                    ticker.c_str(), cb.sigma_V, cdg.lamda, cdg.phi, cdg.V0, cb.rho, cdg.v, cdg.l0);
        // 2. Setup Output File
        std::string out_file = ticker + "pricing.csv";
        std::filesystem::create_directories("./result/");
        std::ofstream data("./result/" + out_file);
        if (!data.is_open()) {
            std::cerr << "Failed to open output file: " << out_file << "\n";
            return;
        }

        data << "ticker,dt,cb_price,default_prob,dp(tree),dp(monte_carlo),default_period\n";

        std::cout << "Pricing " << ticker << "...\n";
        for (double dt : dts) {
            cb.dt_other = dt;
            double dp1 = DefaultTest(cb, cdg, vas, 10000);
            DefaultTestV1Result dp_result = DefaultTestV1(cb, cdg, vas, 10000);
            try {
                FinalResultMemoSave res = CbTreePricingMemoSave(cb, cdg, vas, ticker);
                data << ticker
                    << "," << dt << "," << res.cb_price << "," 
                    << res.default_prob << ","
                    << "," << dp1 << ","
                    << dp_result.average_default_probability
                    << "\n";
            } catch (const std::exception& e) {
                std::cerr << "  -> Error pricing " << ticker << " with dt=" << dt << ": " << e.what() << "\n";
            }
        }
        std::cout << "Finished " << ticker << ". Results saved to " << out_file << "\n\n";
    }
    void bisect_nu_dp(const std::string &ticker, const std::string &file_name, const double target_dp) {
        auto config = factory_functions[ticker]();
        config.cb.sigma_V  = get_market_val(ticker, "sigma_V");
        config.cdg.lamda   =  get_market_val(ticker, "lambda");
        config.cdg.phi     = get_market_val(ticker, "phi");
        config.cdg.V0      = get_market_val(ticker, "V0");
        config.cdg.sigma_v = get_market_val(ticker, "sigma_V");
        config.cb.rho = get_market_val(ticker, "rho");
        config.cdg.v = get_market_val(ticker, "v");
        config.cdg.l0 = std::log(get_market_val(ticker, "debt_ratio_0"));
        config.cb.dt_other = 1.0 / 26.0;

        std::printf("[Info] %s - Injected Market Data: sigma_V=%.4f, lambda=%.4f, phi=%.4f, V0=%.4f, rho=%.4f, v=%.4f, l0=%.4f\n",
                    ticker.c_str(), config.cb.sigma_V, config.cdg.lamda, config.cdg.phi, config.cdg.V0, config.cb.rho, config.cdg.v, config.cdg.l0);
        // 2. Setup Output File
        std::filesystem::create_directories("./result/");
        std::ofstream data("./result/" + file_name);
        if (!data.is_open()) {
            std::cerr << "Failed to open output file: " << file_name << "\n";
            return;
        }
        data << "ticker,cb_price,dp,nu,market_cb\n";
        std::cout << "Pricing " << ticker << "...\n";
        auto fn = [&](double nu) {
            config.cdg.v = nu;
            std::cout << "  -> Testing nu=" << nu << "...\n";
            return DefaultTestV1(config.cb, config.cdg, config.vas, 10000).average_default_probability - target_dp;
        };
        double root_nu = find_root_bisection(fn, 0.0, 10.0, 1e-5);
        config.cdg.v = root_nu;
        FinalResultMemoSave final_res = CbTreePricingMemoSave(config.cb, config.cdg, config.vas, ticker);
        data << ticker << "," << final_res.cb_price
            << ","
            << DefaultTestV1(config.cb, config.cdg, config.vas, 10000).average_default_probability << ","
            << root_nu  << "," 
            << (root_nu != std::nan("") ? cb_market_table[ticker] : -1) << "\n";
        std::cout << "Finished " << ticker << ". Results saved to " << file_name
                  << "\n";
    }
    void bisect_lambda_dp(const std::string &ticker, const std::string &file_name, const double target_dp) {
        auto config = factory_functions[ticker]();
        config.cb.sigma_V  = get_market_val(ticker, "sigma_V");
        config.cdg.lamda   =  get_market_val(ticker, "lambda");
        config.cdg.phi     = get_market_val(ticker, "phi");
        config.cdg.V0      = get_market_val(ticker, "V0");
        config.cdg.sigma_v = get_market_val(ticker, "sigma_V");
        config.cb.rho = get_market_val(ticker, "rho");
        config.cdg.v = get_market_val(ticker, "v");
        config.cdg.l0 = std::log(get_market_val(ticker, "debt_ratio_0"));
        config.cb.dt_other = 1.0 / 24.0;

        std::printf("[Info] %s - Injected Market Data: sigma_V=%.4f, lambda=%.4f, phi=%.4f, V0=%.4f, rho=%.4f, v=%.4f, l0=%.4f\n",
                    ticker.c_str(), config.cb.sigma_V, config.cdg.lamda, config.cdg.phi, config.cdg.V0, config.cb.rho, config.cdg.v, config.cdg.l0);
        // 2. Setup Output File
        std::filesystem::create_directories("./result/");
        std::ofstream data("./result/" + file_name);
        if (!data.is_open()) {
            std::cerr << "Failed to open output file: " << file_name << "\n";
            return;
        }
        data << "ticker,cb_price,dp,lambda,market_cb\n";
        std::cout << "Pricing " << ticker << "...\n";
        auto fn = [&](double lambda) {
            config.cdg.lamda = lambda;
            double prob = DefaultTestV1(config.cb, config.cdg, config.vas, 10000).average_default_probability;
            std::cout << "  -> Testing lambda=" << lambda << "dp: " << prob << "...\n";
            return prob - target_dp;
        };
        double root_lambda = find_root_bisection(fn, 0.001, 10.0);
        config.cdg.lamda = root_lambda;
        FinalResultMemoSave final_res = CbTreePricingMemoSave(config.cb, config.cdg, config.vas, ticker);
        data << ticker << "," << final_res.cb_price
            << ","
            << DefaultTestV1(config.cb, config.cdg, config.vas, 10000).average_default_probability << ","
            << root_lambda  << ","
            << cb_market_table[ticker] << "\n";
        std::cout << "Finished " << ticker << ". Results saved to " << file_name << "\n\n";
    }
};
