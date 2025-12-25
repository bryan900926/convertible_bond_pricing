#include <Eigen/Dense>

#include "HullWhiteModel.h"

PzTreeResult PzTreeBuild(
    int n,
    HullWhiteTreeResult &tree_result)
{
    Eigen::ArrayX3i nxt_idx = Eigen::ArrayX3i::Zero(tree_result.short_rate_tree.rows(), 3);
    const double j_max_other = static_cast<double>(tree_result.j_max_other);
    int start_h = -1;
    for (int j = 0; j < 2 * n + 1; ++j)
    {
        if (tree_result.short_rate_tree(j, 0) != 0.0)
        {
            start_h = j;
        }
        if (j == n - j_max_other)
        {
            nxt_idx(j, 0) = j;
            nxt_idx(j, 1) = j + 1;
            nxt_idx(j, 2) = j + 2;
        }
        else if (j == n + j_max_other)
        {
            nxt_idx(j, 0) = j - 2;
            nxt_idx(j, 1) = j - 1;
            nxt_idx(j, 2) = j;
        }
        else
        {
            nxt_idx(j, 0) = j - 1;
            nxt_idx(j, 1) = j;
            nxt_idx(j, 2) = j + 1;
        }
    }
    return {std::move(nxt_idx), start_h};
}