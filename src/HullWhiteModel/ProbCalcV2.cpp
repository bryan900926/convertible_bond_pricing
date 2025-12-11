#include <Eigen/Dense>
#include <cmath>

#include "HullWhiteModel.h"

Eigen::ArrayX3d ProbCalcV2(int non, int j_max, int n, const Eigen::ArrayX3d &prob_mat)
{
    const int k = n - j_max;
    Eigen::ArrayX3d prob_res = Eigen::ArrayX3d::Zero(2 * (n + 1) + 1, 3);
    const int num_cols = 3;

    if (k > 0)
    {
        const int lim = std::min((int)prob_mat.rows(), k + non);
        prob_res.block(k, 0, lim, num_cols) = prob_mat.block(0, 0, lim, num_cols);
    }
    else
    {
        const int start_a = 1 - (k - 1);
        const int lim = std::min((int)prob_mat.rows() - (start_a - 1), (int)prob_res.rows());
        prob_res.block(0, 0, lim, num_cols) = prob_mat.block(start_a - 1, 0, lim, num_cols);
    }

    return prob_res;
}