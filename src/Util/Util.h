#include "../Pricing/CbModel.h"
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <iostream>

// C++20 Concept: The function 'F' must take a double and return a double
template <typename F>
requires std::is_invocable_v<F, double> && std::same_as<std::invoke_result_t<F, double>, double>
double find_root_bisection(F objective, double left, double right,
                           double tol = 1e-3) {
    double obj_left = objective(left);
    double obj_right = objective(right);
 
    if (obj_left * obj_right > 0) {
        return std::nan(""); // Return Not-a-Number to indicate failure
    }
    
    double mid = 0.0;
    while ((right - left) / 2.0 > tol) {
        mid = left + (right - left) / 2.0;
        double obj_mid = objective(mid); // Exactly ONE tree evaluation per iteration
        
        if (obj_mid == 0.0 || std::abs(obj_mid) < 1e-12) {
            return mid;
        }

        if (obj_left * obj_mid < 0) {
            right = mid;
        }
        else {
            left = mid;
            obj_left = obj_mid; // Update the boundary value without re-evaluating!
        }
    }
    return mid;
}

struct DefaultTestV1Result {
    double average_default_probability;
    double average_default_periods;
};


double DefaultTest(const CbParas &cb_paras, const CdgParas &cdg_paras,
                   const VasciekParas &vasciek_paras, int num_stimulations);

DefaultTestV1Result DefaultTestV1(const CbParas &cb_paras, const CdgParas &cdg_paras,
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