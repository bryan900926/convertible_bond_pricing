#include <Eigen/Dense>
#include <algorithm>

#include "HullWhiteModel.h"

Eigen::ArrayXXd HullWhiteTreeShortRate(const int n, const int j_max_first, const int j_max_other,
                                       const Eigen::ArrayXd &alphas, const Eigen::ArrayXXd &r_tree)
{
    Eigen::ArrayXXd short_rate_tree = Eigen::ArrayXXd::Zero(2 * n + 1, n);

    for (int i = 0; i < n; ++i)
    {
        const int jmax_local = (i == 0) ? j_max_first : j_max_other;
        const int vertical = std::min(jmax_local, i);
        auto row_idx_vec = Eigen::seq(-vertical + n, vertical + n);
        short_rate_tree(row_idx_vec, i) = r_tree(row_idx_vec, i) + alphas(i);
    }

    return short_rate_tree;
}