#pragma once

#include <vector>

#include "..\HullWhiteModel\HullWhiteModel.h"
#include "Eigen/src/Core/Array.h"

struct CbParas {
  double T;
  double sigma_V;
  double F;
  double rr;
  double CR;
  int NS;
  int NC;
  int CP;
  double qdt;
  double rho;
  int partition;
  bool if_const_r;
  double coupon_rate;
  double dt_other;
  int paid_cycle;
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

struct CouponPaidInfo {
  const int total_steps;
  const std::vector<bool> is_coupon_paid;
  const double dt_first;
};

CouponPaidInfo CouponPaidCalc(const double T, const double dt,
                              const int paid_cycle);

void CbTreePricing(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas vasciek_paras);

inline Eigen::ArrayXi
find_indices(const Eigen::Array<bool, Eigen::Dynamic, 1> &mask) {
  Eigen::ArrayXi indices(mask.count());

  int idx = 0;
  for (int i = 0; i < mask.size(); ++i) {
    if (mask(i))
      indices(idx++) = i;
  }

  return indices;
}

struct EquityTreeBuildResult;

void CbTreeBuild(const CbParas &cb_paras, const CdgParas &cdg_paras,
                 const VasciekParas &vasciek_paras,
                 const EquityTreeBuildResult &equity_tree_result,
                 const HullWhiteTreeResult &tree_result,
                 const CouponPaidInfo &coupon_info,
                 const PzTreeResult &pz_result,
                 const std::vector<int> &num_node_steps);

void FillMap(Eigen::ArrayXi &map, const int step,
             const Eigen::ArrayX3i &idx_vec,
             const std::vector<int> &cum_node_steps,
             const std::vector<int> &num_node_steps, const int m_idx_offset,
             const int cols);

Eigen::ArrayXd DoInterp(const Eigen::ArrayXd &val1, const Eigen::ArrayXd &val2,
                        const Eigen::ArrayXd &weight);

void CbTreeBuildMemoSave(const CbParas &cb_paras, const CdgParas &cdg_paras,
                         const VasciekParas &vasciek_paras,
                         const EquityTreeBuildResult &equity_tree_result,
                         const HullWhiteTreeResult &tree_result,
                         const CouponPaidInfo &coupon_info,
                         const PzTreeResult &pz_result,
                         const std::vector<int> &num_node_steps);