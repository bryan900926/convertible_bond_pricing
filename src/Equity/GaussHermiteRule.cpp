#include <cmath>
#include <Eigen/Dense>

#include "EquityModel.h"

// Constants
const double PI = 3.14159265358979323846;

GaussHermiteResult compute_gauss_hermite_rule(int n)
{
    GaussHermiteResult res;
    // 1. Construct the symmetric Companion Matrix (Tridiagonal)
    // MATLAB: i = 1:n-1; a = sqrt(i/2);
    Eigen::MatrixXd CM = Eigen::MatrixXd::Zero(n, n);

    for (int i = 0; i < n - 1; ++i)
    {
        // i corresponds to MATLAB index i+1
        double val = std::sqrt((double)(i + 1) / 2.0);

        // Fill super-diagonal (i, i+1) and sub-diagonal (i+1, i)
        CM(i, i + 1) = val;
        CM(i + 1, i) = val;
    }

    // 2. Solve Eigenvalue Problem
    // MATLAB: [V, L] = eig(CM);
    // SelfAdjointEigenSolver is optimized for symmetric matrices
    // and returns eigenvalues sorted in ascending order.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(CM);

    if (solver.info() != Eigen::Success)
    {
        // Handle error (throw exception or return empty)
        return res;
    }

    // 3. Extract Nodes (Eigenvalues)
    // MATLAB: [x, ind] = sort(diag(L));
    res.x = solver.eigenvalues().array();

    // 4. Extract Weights
    // MATLAB: w = sqrt(pi) * V(:, 1).^2 (after transposing/sorting)
    // The theory states: w_j = sqrt(pi) * (first component of j-th normalized eigenvector)^2
    // Eigen stores eigenvectors in columns. We want the first row.
    const double SQRT_PI = 1.77245385090551602729;

    // .row(0) gives the first component of all eigenvectors
    // .array().square() squares them element-wise
    res.w = SQRT_PI * solver.eigenvectors().row(0).array().square();

    return res;
}