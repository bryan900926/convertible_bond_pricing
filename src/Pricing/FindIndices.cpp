#include <Eigen/Dense>

Eigen::ArrayXi FindIndices(const Eigen::Array<bool, Eigen::Dynamic, 1> &mask) {
  Eigen::ArrayXi indices(mask.count());

  int idx = 0;
  for (int i = 0; i < mask.size(); ++i) {
    if (mask(i))
      indices(idx++) = i;
  }

  return indices;
}
