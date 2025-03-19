#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

Eigen::MatrixXd invertMatrix(const Eigen::MatrixXd& input_matrix) {
    // Attempt to perform LU decomposition with partial pivoting
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(input_matrix);

    // Check if the LU decomposition succeeded (matrix is non-singular)
    if (lu_decomp.isInvertible()) {
        // Calculate the inverse using LU decomposition
        Eigen::MatrixXd inverted_matrix = lu_decomp.inverse();
        return inverted_matrix;
    } else {
        // Matrix is singular; calculate the pseudo-inverse
        Eigen::MatrixXd pseudo_inverse = input_matrix.completeOrthogonalDecomposition().pseudoInverse();
        return pseudo_inverse;
    }
}

namespace py = pybind11;

PYBIND11_MODULE(matrix_inversion, m) {
    m.def("invert_matrix", &invertMatrix, "Invert a matrix or calculate its pseudo-inverse");
}
