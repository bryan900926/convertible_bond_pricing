#include "HullWhiteModel.h"
#include <vector>
#include <Eigen/Dense>

Eigen::ArrayX3d ProbCalc(int non, int j_max, double dt, double kappa)
{
    Eigen::ArrayX3d probs(non, 3);

    Eigen::ArrayXd j_vec = Eigen::ArrayXd::LinSpaced(non, -j_max, j_max);
    Eigen::ArrayXd idx_vec = j_vec + j_max;
    Eigen::ArrayXd nu = kappa * dt * j_vec;
    Eigen::ArrayXd nu2 = nu.square();

    const double one_sixth = 1.0 / 6.0;
    const double seven_sixths = 7.0 / 6.0;
    const double two_thirds = 2.0 / 3.0;
    const double one_third = 1.0 / 3.0;

    // inner points
    probs.col(0) = one_sixth + 0.5 * (nu2 - nu);
    probs.col(1) = two_thirds - nu2;
    probs.col(2) = one_sixth + 0.5 * (nu2 + nu);

    // min bound
    probs(0, 0) = one_sixth + 0.5 * (nu2(0) + nu(0));
    probs(0, 1) = -one_third - nu2(0) - 2 * nu(0);
    probs(0, 2) = seven_sixths + 0.5 * (nu2(0) + 3 * nu(0));

    // max bound
    probs(non - 1, 0) = seven_sixths + 0.5 * (nu2(non - 1) - 3 * nu(non - 1));
    probs(non - 1, 1) = -one_third - nu2(non - 1) + 2 * nu(non - 1);
    probs(non - 1, 2) = one_sixth + 0.5 * (nu2(non - 1) - nu(non - 1));

    return probs; //probs = [pu_vec, pm_vec, pd_vec]
}