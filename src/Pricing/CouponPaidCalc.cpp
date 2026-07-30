#include <cmath>
#include <vector>
#include <utility>

#include "CbModel.h"

constexpr double EPSILON = 1e-9;
/// @brief Calculate the coupon payment schedule for a convertible bond. Since at most of starting point,
/// it might be in the middle of the coupon payment period, we need to calculate the first step separately.
/// @param T Total time to maturity
/// @param dt time step in after the first step, the first step is different from the rest of the steps, so we need to calculate the first step separately
/// @param paid_cycle The period of coupon payments
/// @return A CouponPaidInfo structure containing the coupon payment schedule
CouponPaidInfo CouponPaidCalc(
    const double T,
    const double dt,
    const double paid_cycle)
{
    int n = static_cast<int>(std::floor((T / dt) + EPSILON));

    std::vector<bool> is_coupon_paid(n + 2, false);

    double first_dt = T - n * dt;

    if (first_dt >= 1e-5)
    {
        n++;
    }
    else
    {
        first_dt = dt;
    }

    if (n >= is_coupon_paid.size())
    {
        is_coupon_paid.resize(n + 1);
    }

    is_coupon_paid[n] = true;

    int cnt = 1;
    for (int i = n - 1; i >= 1; --i)
    {
        if ((cnt * dt) >= (paid_cycle - EPSILON))
        {
            is_coupon_paid[i] = true;
            cnt = 0;
        }
        cnt++;
    }

    return {n, std::move(is_coupon_paid), first_dt};
}