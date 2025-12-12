#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "src/HullWhiteModel/HullWhiteModel.h"

int main()
{
    auto start = std::chrono::steady_clock::now();
    Eigen::ArrayXd time_grid = Eigen::ArrayXd::LinSpaced(10, 0.1, 1);
    double dt_first = time_grid(0), dt_other = time_grid(1) - time_grid(0), sigma_r = 0.01, kappa = 0.1, r_bar = 0.05, r0 = 0.03;
    Eigen::ArrayXd zero_rates = VasciekZeroRates({kappa, r_bar, sigma_r, r0}, time_grid);
    auto tree =  HullWhiteTree(kappa, sigma_r, zero_rates, dt_first, dt_other);
    std::cout << "Hull-White Short Rate Tree:\n" << tree << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    return 0;
}