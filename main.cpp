#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "src/HullWhiteModel/HullWhiteModel.h"

int main()
{
    double dt_first = 0.1, dt_other = 0.1, sigma_r = 0.01, kappa = 0.1, r_bar = 0.05, r0 = 0.03;
    Eigen::ArrayXd zero_rates = VasciekZeroRates({kappa, r_bar, sigma_r, r0}, Eigen::ArrayXd::LinSpaced(10, 0.1, 1));
    auto tree =  HullWhiteTree(kappa, sigma_r, zero_rates, dt_first, dt_other);
    std::cout << "Hull-White Short Rate Tree:\n" << tree << std::endl;
    return 0;
}