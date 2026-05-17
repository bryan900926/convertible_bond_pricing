#include "Pricing/CbModel.h"
#include <vector>
#include <string>
#include <cmath>
#include <climits>

enum class PricingMethod {
    DP_BISECTION = 1 << 0, // change the parameter like lambda or v, and find the root of the equation dp - dp_market = 0
    PRICING = 1 << 1, // just run the pricing with given parameters
};

struct PricingConfig {
    std::string name;
    CbParas cb;
    CdgParas cdg;
    VasciekParas vas;
    int methods; // bitmask of PricingMethod
};

class ConfigFactory {
public:
    static PricingConfig get_LAB() {
        std::vector<CallInfo> calls = {CallInfo("2016-02-01", 130.0), CallInfo("2021-02-05", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2016-02-01");
        
        return {
          "LAB",
              {18, 0, 100, 0.4954, 1.78750, 28844000, 2013000, 100,
               20, 0, 20,  false,  0.0275,  1.0 / 12, 1,       call_schedule},
              {0.7, 0, std::log(0.45101481855527364), 0, 0.010226007, 0, 1.81},
              {0.136179596167386, 0.0447875855396496, 0.0200316752383784,
               0.00393433054895577},
        };
    }

    static PricingConfig get_EEFT() {
        std::vector<CallInfo> calls = {CallInfo("2022-09-20", 130.0), CallInfo("2025-03-20", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2021-03-15");
        
        return {
            "EEFT",
            { 28, 0, 100, 0.4456, 0.52987, 52753000, 4522280, INT_MAX, 20, 0, 20, false, 0.0075, 1.0/6, 1, call_schedule},
            { 0.7, 0, std::log(0.28741741636296036), 0, 0.003567945, 0, 1.81 },
            { 0.22336291527447, 0.0339962229038466, 0.0381091909084148, 0.000879559544106292 },
        };
    }

    static PricingConfig get_DHR() {
        std::vector<CallInfo> calls = {
            CallInfo("2010-01-22", 77.128), CallInfo("2011-01-22", 78.970),
            CallInfo("2012-01-22", 80.857), CallInfo("2013-01-22", 82.789),
            CallInfo("2014-01-22", 84.767), CallInfo("2015-01-22", 86.792),
            CallInfo("2016-01-22", 88.865), CallInfo("2017-01-22", 90.989),
            CallInfo("2018-01-22", 93.162), CallInfo("2019-01-22", 95.388),
            CallInfo("2020-01-22", 97.667)
        };
        CallSchedule call_schedule = CallSchedule::Create(calls, "2009-01-22");
        
        return {
            "DHR",
            { 12.0, 0, 100, 0.4954, 1.45352, 354487000, 6200000, 771, 20, 0, 20, false, 0.0, 1.0/12, 1, call_schedule },
            { 0.7, 0, std::log(0.39883480752180996), 0, 0.004711, 0, 1.81 },
            { 0.386248236006081, 0.0469941534383253, 0.0593403428043975, 0.00260058521741104 },
        };
    }

    static PricingConfig get_COLL() {
        std::vector<CallInfo> calls = {CallInfo("2023-02-15", 130.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2021-02-16");
        
        return {
            "COLL",
            { 7.0, 0, 100, 0.4456, 3.67815, 34756000, 1437500, 130, 20, 0, 20, false, 0.02625, 1.0/12, 1, call_schedule},
            { 1.7, 0, std::log(0.357547605), 0, 0.025091036, 0, 1 },
            { 0.01628559909171213, 0.1595220619446585, 0.00428873644512, 0.0009307909267961937 },
        };
    }

    static PricingConfig get_EXPE() {
        std::vector<CallInfo> calls = {CallInfo("2024-02-20", 130.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-02-15");
        
        return {
            "EXPE",
            { 4.0, 0, 100, 0.4456, 0.39212, 150231000, 986000, 1300, 20, 0, 20, false, 0.0, 1.0/12, 1, call_schedule},
            { 1.7, 0, std::log(0.9060275546106771), 0, 0.009263, 0, 1 },
            { 0.04205976155550541, 0.283762364578435, 0.0221521133952676, 0.00427564701108187 },
        };
    }

    static PricingConfig get_SPOT() {
        std::vector<CallInfo> calls = {CallInfo("2024-03-20", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-02-15");
        
        return {
            "SPOT",
            { 4.0, 0, 100, 0.4456, 0.1941, 192152000, 1500000, 130, 20, 0, 20, false, 0.0, 1.0/12, 1, call_schedule },
            { 4.9, 0, std::log(0.15723057959785527), 0, 0.001669371, 0, 1.81 },
            { 0.136179596167386, 0.0447875855396496, 0.0831316482986798, 0.00393433054895577 },
        };
    }

    static PricingConfig get_FSLY() {
        std::vector<CallInfo> calls = {CallInfo("2024-03-20", 130.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-03-15");
        
        return {
            "FSLY",
            { 4.0, 0, 100, 0.4456, 0.97272, 117500000, 9487500, 130, 20, 0, 20, false, 0.0, 1.0/12, 1, call_schedule },
            { 0.7, 0, std::log(0.3554711592587863), 0, 0.001910412, 0, 1.81 },
            { 2.73186466686729, 0.0557494010613996, 0.120590482306319, 0.0326238320383245 },
        };
    }

    static PricingConfig get_VERI() {
        std::vector<CallInfo> calls = {CallInfo("2024-11-20", 100.0)};
        CallSchedule call_schedule = CallSchedule::Create(calls, "2022-11-15");
        
        return {
            "VERI",
            { 4.0, 0, 100, 0.4456, 2.72068, 36290000, 2013000, 130, 20, 0, 20, false, 0.0175, 1.0/12, 1, call_schedule },
            { 2.1, 0, std::log(0.6611612151495524), 0, 0.006972534, 0, 0.21 },
            { 2.73186466686729, 0.0557494010613996, 0.120590482306319, 0.0326238320383245 },
        };
    }
};