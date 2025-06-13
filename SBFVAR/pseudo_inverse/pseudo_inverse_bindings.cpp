#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

Eigen::MatrixXd calculatePseudoInverse(const Eigen::MatrixXd &matrix);

namespace py = pybind11;

PYBIND11_MODULE(pseudo_inverse, m) {
    m.def("calculate_pseudo_inverse", &calculatePseudoInverse);
}
