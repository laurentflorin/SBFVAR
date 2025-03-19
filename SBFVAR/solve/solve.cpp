#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

Eigen::MatrixXd linalg_solve(Eigen::MatrixXd A, Eigen::MatrixXd B) {
    Eigen::MatrixXd X = A.colPivHouseholderQr().solve(B);
    return X;
}

PYBIND11_MODULE(linalg_solve, m) {
    m.def("linalg_solve", &linalg_solve, "Solve a system of linear equations using Eigen.");
}
