#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "cholcov.h"

namespace py = pybind11;

PYBIND11_MODULE(cholcov_module, m) {
    m.doc() = "Cholesky or Cholesky-like decomposition function";
    
    m.def("cholcovOrEigendecomp", &cholcovOrEigendecomp, "Compute Cholesky or Cholesky-like decomposition of a matrix");
}
