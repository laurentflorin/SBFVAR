#include <Eigen/Core>
#include <Eigen/Dense>

Eigen::MatrixXd calculatePseudoInverse(const Eigen::MatrixXd &matrix) {
    return matrix.completeOrthogonalDecomposition().pseudoInverse();
}
