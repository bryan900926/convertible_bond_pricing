#include "../Equity/EquityModel.h"

double Calc_Bk(double k, double t) {
  double u = k * t;
  if (std::abs(u) < 1e-12)
    return t;
  return -std::expm1(-u) / k;
}