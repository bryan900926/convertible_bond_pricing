#include <Eigen/Dense>
#include <vector>


void FillMap(Eigen::ArrayXi &map, const int step,
              const Eigen::ArrayX3i &idx_vec,
              const std::vector<int> &cum_node_steps,
              const std::vector<int> &num_node_steps, const int m_idx_offset,
              const int cols) {
  const auto m_arr =
      idx_vec.col(1).middleRows(cum_node_steps[step - 1], num_node_steps[step]);
  const auto k_arr =
      idx_vec.col(2).middleRows(cum_node_steps[step - 1], num_node_steps[step]);

  const auto idxs = (m_arr + m_idx_offset) * cols + k_arr;
  const Eigen::ArrayXi target_idx =
      Eigen::ArrayXi::LinSpaced(m_arr.size(), 0, m_arr.size() - 1);
  map(idxs) = target_idx;
};