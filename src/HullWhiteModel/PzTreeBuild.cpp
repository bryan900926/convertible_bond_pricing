#include <Eigen/Dense>
#include "HullWhiteModel.h"

/// @brief This function builds the PzTreeResult from the HullWhiteTreeResult. 
/// It constructs the next index tree (nxt_idx) based on the short rate tree and the maximum jump indices (j_max_other). 
/// The function also identifies the starting index (start_h) for the short rate tree.
/// @param n 
/// @param tree_result 
/// @return 
PzTreeResult PzTreeBuild(
    int n,
    HullWhiteTreeResult &tree_result)
{
    Eigen::ArrayX3i nxt_idx = Eigen::ArrayX3i::Zero(tree_result.short_rate_tree.rows(), 3);
    const double j_max_other = static_cast<double>(tree_result.j_max_other);
    int start_h = -1;
    // std::cout << "Row: " << tree_result.short_rate_tree.rows() << ", Col: " << tree_result.short_rate_tree.cols() << std::endl;

    for (int j = 0; j < nxt_idx.rows(); ++j)
    {
        if (tree_result.short_rate_tree(j, 0) != 0.0)
        {
            start_h = j;
        }
    }
    for (int j = 0; j < nxt_idx.rows(); ++j) {
        if (j <= start_h - j_max_other) {
            nxt_idx(j, 0) = j;
            nxt_idx(j, 1) = j + 1;
            nxt_idx(j, 2) = j + 2;
        }
        else if (j >= start_h + j_max_other)
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