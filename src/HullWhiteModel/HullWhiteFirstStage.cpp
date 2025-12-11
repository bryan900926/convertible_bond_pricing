#include <Eigen/Dense>
#include <cmath>

Eigen::ArrayXXd HullWhiteFirstStage(
    const int n, const double sigma_r, const int j_max_first,
    const int j_max_other, const double dt_first, const double dt_other)
{
    Eigen::ArrayXXd hw_first_stage = Eigen::ArrayXXd::Zero(2 * n + 1, n);
    const double dr_first = sigma_r * std::sqrt(3 * dt_first);
    const double dr_other = sigma_r * std::sqrt(3 * dt_first);

    for (int i = 1; i < n; ++i)
    {
        const int j_max = i == 1 ? j_max_first : j_max_other;
        const int dr = i == 1 ? dr_first : dr_other;
        const int vertical = std::min(i, j_max);
        for (int j = -vertical; j <= vertical; ++j)
        {
            hw_first_stage(j + n, i) = j * dr;
        }
    }

    return hw_first_stage;
}
