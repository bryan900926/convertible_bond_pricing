#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "src/HullWhiteModel/HullWhiteModel.h"
#include "src/Pricing/CbModel.h"
#include "src/Equity/EquityModel.h"
#include "src/Util/Timer.h"

int main()
{
    Timer timer("CbTreePricing Main Timer");
    CbParas cb_paras = {
        1.0,     // T
        0.1,     // sigma_V
        100.0,   // F
        0.4,     // rr
        0.05,    // CR
        36000,   // NS
        100,     // NC
        1,       // CP
        20,      // qdt
        0,       // rho
        20,      // partition
        false,  // if_const_r
        0.06,    // coupon_rate
        1.0 / 144, // dt_other
        1        // paid_cycle
    };
    CdgParas cdg_paras = {
        0.18,    // lamda
        0,       // phi
        -0.3,    // l0
        4550000, // V0
        0.005,   // delta
        0.1,     // sigma_v
        0.6,     // v
    };
    VasciekParas vasciek_paras = {
        0.1,  // kappa
        0.05, // r_bar
        0.01, // sigma_r
        0.05  // r0
    };

    CbTreePricing(
        cb_paras,
        cdg_paras,
        vasciek_paras);
    return 0;
}