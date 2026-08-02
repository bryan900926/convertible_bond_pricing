# Convertible Bond Pricing Model

## ⚠️ Prerequisites & Compiler Notes

*   **Build Tool:** [CMake](https://cmake.org/download/) is required to configure and build the project.
*   **Recommended Compiler:** `gcc version 16.1.0 (Rev1, Built by MSYS2 project)`.
*   **Warning on Compilers & Optimization:** It is highly recommended to stick to the tested compiler version. Older compilers, as well as overly aggressive optimization flags (such as `-O3`), are known to cause unexpected floating-point arithmetic exceptions and crashes during runtime.

---

## 🚀 How to Build and Run

Open your terminal, navigate to the root directory of the project, and run the following commands based on your desired build profile.

**For a Release Build (Optimized for performance):**
```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build_release
```

**For a Debug Build (For testing and debugging):**
```bash
cmake -S . -B build_debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++
cmake --build build_debug
```

*(Note: After the final `--build` command finishes, your compiled executable will be located inside the respective `build_release` or `build_debug` folder).*

---

## 📁 Project Structure

| Directory         | Description                                                                                                                                       |
| :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/Pricing/`    | **Core:** Contains the primary convertible bond pricing logic.                                                                                    |
| `src/Equity/`     | Calculates the equity prices used within the CB pricing models.                                                                                   |
| `src/Simulation/` | Monte Carlo simulations used to approximate the default probability of a company.                                                                 |
| `temp_data/`      | **Disk Caching:** Due to RAM limitations, tree data at each period is written to disk here. *(See `TreeManager.hpp` for implementation details).* |
| `src/Script/`     | PowerShell scripts designed to automatically run multiple programs/tasks at once.                                                                 |
| `src/Legacy/`     | Old versions of certain functions. You can generally ignore this folder.                                                                          |

---

## 📊 Data Input

The program expects the following configuration files for execution:

*   **`schedule_para.csv`**: Contains the convertible bond schedules. All target tickers must be written into this file.
*   **`para_test.csv`**: Contains all other underlying model parameters.
