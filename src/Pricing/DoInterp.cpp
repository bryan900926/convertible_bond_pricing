#include <Eigen/Dense>

Eigen::ArrayXd DoInterp(const Eigen::ArrayXd &val1,
                             const Eigen::ArrayXd &val2,
                             const Eigen::ArrayXd &weight) {
  return val1 + weight * (val2 - val1);
}