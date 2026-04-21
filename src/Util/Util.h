#include <concepts>
#include <cmath>
#include <iostream>

// C++20 Concept: The function 'F' must take a double and return a double
template <typename F>
  requires std::invocable<F, double> &&
           std::same_as<std::invoke_result_t<F, double>, double>
double find_root_bisection(F objective, double left, double right,
                           double tol = 1e-3) {
    double obj_left = objective(left);
    double obj_right = objective(right);

    // 1. Safety Check: Ensure the root is actually bracketed
    if (obj_left * obj_right > 0) {
        std::cerr << "Error: Root is not bracketed in [" << left << ", " << right << "]\n";
        return std::nan(""); // Return Not-a-Number to indicate failure
    }
    // Basic bisection implementation
    double mid = 0.0;
    while ((right - left) / 2.0 > tol) {
      mid = left + (right - left) / 2.0;
      std::cout << "Left: " << left << ", Right: " << right << ", Mid: " << mid << "\n";
      double obj_mid = objective(mid);
      if (obj_mid == 0.0) return mid;

      if (objective(left) * obj_mid < 0) {
          right = mid;
      } else {
            left = mid;
        }
    }
    return mid;
}