#pragma once

#include <chrono>
#include <vector>

#include "..\HullWhiteModel\HullWhiteModel.h"

class CallInfo {
public:
  double call_price;
  double digit_date; // days from pricing date, to be calculated during initialization
  std::string date_str;
  CallInfo(const std::string& date_str, double call_price)
      : date_str(date_str), call_price(call_price) {}
};

class CallSchedule {
    std::vector<CallInfo> _call_infos;
    int _current_index = 0;
    std::string _pricing_date;
    // Helper to convert "YYYY-MM-DD" strings into exact day differences
    static double CalculateDaysBetween(const std::string& pricing_date, const std::string& target_date) {
        auto parse_date = [](const std::string& date_str) {
            int y, m, d;
            sscanf(date_str.c_str(), "%d-%d-%d", &y, &m, &d);
            return std::chrono::sys_days{std::chrono::year{y} / m / d};
        };

        auto start_sys = parse_date(pricing_date);
        auto target_sys = parse_date(target_date);

        return (target_sys - start_sys).count() / 365.0; // Convert days to years
    }
    CallSchedule(const std::vector<CallInfo>& calls, const std::string& pricing_date)
        : _call_infos(calls), _pricing_date(pricing_date) {
    }
public:
    static CallSchedule Create(const std::vector<CallInfo>& calls, const std::string& pricing_date) {
        std::vector<CallInfo> mutable_calls;
        
        for (auto& call : mutable_calls) {
          call.digit_date = CalculateDaysBetween(pricing_date, call.date_str);
          // filter out calls that are after the pricing date
          if (call.digit_date >= 0) {
              mutable_calls.push_back(call);
          }
        }
        std::sort(mutable_calls.begin(), mutable_calls.end(), 
            [](const CallInfo& a, const CallInfo& b) {
                return a.digit_date < b.digit_date;
            });

        return CallSchedule(std::move(mutable_calls), pricing_date); 
    }
    /// @brief This method assume company can call multiple times, and return the latest call price if the current time is after the call date
    /// @param current_time_in_days 
    /// @return 
    double GetActiveCallPrice(double current_time_in_days) const {
        
        double active_price = std::numeric_limits<double>::max();

        for (const auto& call : _call_infos) {
            if (current_time_in_days >= call.digit_date) {
                active_price = call.call_price;
            } 
        }
        return active_price;
    }
    /// @brief This method assume investor can only call once
    /// @param current_time_in_days 
    /// @return 
    double GetActiveCallOneTime(double current_time_in_days) {
        if (_current_index < _call_infos.size() && current_time_in_days >= _call_infos[_current_index].digit_date) {
            return _call_infos[_current_index++].call_price;
        }
        return std::numeric_limits<double>::max();
    }
};
struct CbParas {
  double T;
  double F;
  double rr;
  double CR;
  double NS;
  int NC;
  int CP;
  double qdt;
  double rho;
  int partition;
  bool if_const_r;
  double coupon_rate;
  double dt_other;
  double paid_cycle;
  double s0;
  mutable CallSchedule call_schedule;
};

struct CdgParas {
  double lamda;
  double phi;
  double l0;
  double V0;
  double delta;
  double sigma_v;
  double v;
};

struct LNode {
  size_t step;
  size_t k;
  int m;
  double l_min;
  double l_max;
  double cum_prob = 1;
};

struct PNode {
  size_t step;
  size_t k;
  int m;
  double prob_matrix[3][3];
};

struct MNode {
  size_t step;
  size_t k;
  int m;
  int nxt_m;
};

struct PackedNode {
  int step;
  size_t k;
  int m;
  double l_min;
  double l_max;
  double p_x_up;
  double p_x_mid;
  double p_x_down;
  int nxt_middle_m;
};

struct CouponPaidInfo {
  const int total_steps;
  const std::vector<bool> is_coupon_paid;
  const double dt_first;
};
struct FinalResult {
  double cb_price;
  double default_prob;
  double equity_forward_price;
  double equity_backward_price;
};

struct FinalResultMemoSave {
  double cb_price;
  double default_prob;
  double zcb_price;
  bool convert_at_t0;
};

CouponPaidInfo CouponPaidCalc(const double T, const double dt,
                              const double paid_cycle);

FinalResult CbTreePricing(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas vasciek_paras);

FinalResultMemoSave CbTreePricingMemoSave(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas& vasciek_paras, const std::string& ticker);

struct EquityTreeBuildResult;

FinalResult CbTreeBuildV2(const CbParas &cb_paras, const CdgParas &cdg_paras,
                          const VasciekParas &vasciek_paras,
                          const EquityTreeBuildResult &equity_tree_result,
                          const HullWhiteTreeResult &tree_result,
                          const CouponPaidInfo &coupon_info,
                          const PzTreeResult &pz_result,
                          const std::vector<int> &num_node_steps);

class TreeManager;

FinalResultMemoSave CbTreeBuildMemoSave(const CbParas &cb_paras,
                                        const CdgParas &cdg_paras,
                                        const VasciekParas &vasciek_paras,
                                        const HullWhiteTreeResult &tree_result,
                                        const CouponPaidInfo &coupon_info,
                                        const PzTreeResult &pz_result,
                                        const int m_idx_offset,
                                        TreeManager &tree_manager
);
