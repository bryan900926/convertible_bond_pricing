#include "Eigen/Dense"

#include "..\Pricing\CbModel.h"
#include "..\HullWhiteModel\HullWhiteModel.h"
#include "EquityModel.h"

Eigen::ArrayXXd EquityFunVec(
    const CbParas &cb_paras,
    const EquityContext &ctx)
{
    GaussHermiteResult gh_result = compute_gauss_hermite_rule(cb_paras.qdt);
    Eigen::ArrayXXd e_arr = Eigen::ArrayXXd::Zero(ctx.m_arr.rows(), cb_paras.partition);
    for (size_t k = 0; k < cb_paras.qdt; ++k)
    {
        e_arr += gh_result.w(k) * FormulaFastVec(
                                      gh_result.x(k),
                                      ctx);
    }
    return e_arr;
}