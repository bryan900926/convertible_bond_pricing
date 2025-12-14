#include <Eigen/Dense>
#include <vector>
#include <algorithm> // For std::clamp or min/max

#include "HullWhiteModel.h"


PzTreeResult PzTreeBuild(
    const Eigen::ArrayXXd &hw_tree,
    int n,
    int jmax_other)
{
    const int grid_height = static_cast<int>(hw_tree.rows());

    PzTreeResult result;
    result.nxt_idx_mat.resize(n); // Allocate vector slots for time steps

    Eigen::ArrayXi j_vec = Eigen::ArrayXi::LinSpaced(grid_height, 0, grid_height - 1);

    for (int i = 0; i < n; ++i)
    {
        result.nxt_idx_mat[i].resize(grid_height, 3);

        Eigen::ArrayX3i &step_mat = result.nxt_idx_mat[i];

        step_mat.col(0) = j_vec - 1;
        step_mat.col(1) = j_vec;
        step_mat.col(2) = j_vec + 1;

        int upper_idx = std::min(grid_height - 1, n + jmax_other); // Safety clamp
        if (upper_idx < grid_height)
        {
            step_mat(upper_idx, 0) = upper_idx - 2;
            step_mat(upper_idx, 1) = upper_idx - 1;
            step_mat(upper_idx, 2) = upper_idx;
        }

        int lower_idx = std::max(0, n - jmax_other); // Safety clamp
        if (lower_idx >= 0)
        {
            step_mat(lower_idx, 0) = lower_idx;
            step_mat(lower_idx, 1) = lower_idx + 1;
            step_mat(lower_idx, 2) = lower_idx + 2;
        }
    }

    for (int j = 0; j < grid_height; ++j)
    {
        if (hw_tree(j, 0) != 0)
        {
            result.start_h = j;
            break;
        }
    }

    return result; // RVO (Return Value Optimization) makes this copy-free
}