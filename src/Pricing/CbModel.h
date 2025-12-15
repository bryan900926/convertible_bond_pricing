#pragma once

#include <vector>

#include "..\HullWhiteModel\HullWhiteModel.h"

struct CbParas
{
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

struct CdgParas
{
    double lamda;
    double phi;
    double l0;
    double V0;
    double delta;
    double sigma_v;
    double v;
};


struct CouponPaidInfo
{
    const int total_steps;
    const std::vector<bool> is_coupon_paid;
    const double dt_first;
};

CouponPaidInfo CouponPaidCalc(
    const double T,
    const double dt,
    const int paid_cycle);

void CbTreePricing(const CbParas &cb_paras, const CdgParas &cdg_paras, const VasciekParas vasciek_paras);