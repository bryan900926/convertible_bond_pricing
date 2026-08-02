#pragma once

#include <chrono>
#include <vector>

#include "..\HullWhiteModel\HullWhiteModel.h"

class ScheduleInfo {
public:
  enum class ScheduleType { Call, Put };
  double price;
  double digit_date; // days from pricing date, to be calculated during initialization
  std::string date_str;
  ScheduleType type;

  ScheduleInfo(const std::string& date_str, double price, ScheduleType type)
      : date_str(date_str), price(price), type(type) {}
};

class CallSchedule {
    std::vector<ScheduleInfo> _infos;
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
    CallSchedule(const std::vector<ScheduleInfo>& schedules, const std::string& pricing_date)
        : _infos(schedules), _pricing_date(pricing_date) {
    }
public:
    static CallSchedule Create(const std::vector<ScheduleInfo>& schedules, const std::string& pricing_date) {
        std::vector<ScheduleInfo> mutable_schedules = schedules;

        for (auto& schedule : mutable_schedules) {
            schedule.digit_date = CalculateDaysBetween(pricing_date, schedule.date_str);
            // filter out schedules that are after the pricing date
            if (schedule.digit_date >= 0) {
              mutable_schedules.push_back(schedule);
          }
        }
        std::sort(mutable_schedules.begin(), mutable_schedules.end(), 
            [](const ScheduleInfo& a, const ScheduleInfo& b) {
                return a.digit_date < b.digit_date;
            });

        return CallSchedule(std::move(mutable_schedules), pricing_date); 
    }
    /// @brief This method assume investor can only excercise the option once
    /// @param current_time_in_years
    /// @return
    double CheckIfExcerciseOneTime(double current_time_in_years, double current_cb_value) {
        if (_current_index < _infos.size() && current_time_in_years >= _infos[_current_index].digit_date) {
          double active_price = _infos[_current_index].price;
          ScheduleInfo::ScheduleType type = _infos[_current_index].type;
          if (type == ScheduleInfo::ScheduleType::Call && current_cb_value > active_price) {
              return active_price;
          } else if (type == ScheduleInfo::ScheduleType::Put && current_cb_value < active_price) {
              return active_price;
          }
          _current_index++; // Move to the next schedule for future scheduling
        }
        return current_cb_value;
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
};

CouponPaidInfo CouponPaidCalc(const double T, const double dt,
                              const double paid_cycle);

FinalResult CbTreePricing(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas vasciek_paras);

FinalResultMemoSave CbTreePricingMemoSave(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas& vasciek_paras, const std::string& ticker);

struct EquityTreeBuildResult;

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
