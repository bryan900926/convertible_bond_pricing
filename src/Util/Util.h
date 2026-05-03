#include <concepts>
#include "../Pricing/CbModel.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <type_traits>
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

double DefaultTest(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas &vasciek_paras, int num_stimulations);

double DefaultTestV1(const CbParas &cb_paras, const CdgParas &cdg_paras,
                     const VasciekParas &vasciek_paras, int num_stimulations);

namespace fs = std::filesystem;

template <typename T>
inline void saveData(const std::vector<T>& data, const std::string& filename) {
    // 1. Safeguard: Ensure type is safe for raw memory copying
    static_assert(std::is_trivially_copyable<T>::value, "Type T must be trivially copyable for binary I/O");

    // 2. Ensure directory exists
    fs::path dir("temp_cache");
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }

    std::ofstream ofs(dir / filename, std::ios::binary | std::ios::out);
    if (!ofs) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    size_t num_elements = data.size();
    ofs.write(reinterpret_cast<const char*>(&num_elements), sizeof(size_t));

    if (num_elements > 0) {
        ofs.write(reinterpret_cast<const char*>(data.data()), num_elements * sizeof(T));
    }
}

template <typename T>
inline std::vector<T> loadData(const std::string& filename) {
    static_assert(std::is_trivially_copyable<T>::value, "Type T must be trivially copyable for binary I/O");

    std::ifstream ifs("temp_cache/" + filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    size_t num_elements = 0;
    ifs.read(reinterpret_cast<char*>(&num_elements), sizeof(size_t));

    // Basic integrity check: check if file is at least large enough to hold the data
    std::vector<T> data(num_elements);
    if (num_elements > 0) {
        ifs.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    }
    return data;
}