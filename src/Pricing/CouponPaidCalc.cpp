#include <cmath>
#include <vector>
#include <utility>

#include "CbModel.h"

constexpr double EPSILON = 1e-9;

CouponPaidInfo CouponPaidCalc(
    const double T,
    const double dt,
    const int paid_cycle)
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