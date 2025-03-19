#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXd cholcovOrEigendecomp(const Eigen::MatrixXd& cov_matrix) {
    Eigen::LLT<Eigen::MatrixXd> llt(cov_matrix);

    if (llt.info() == Eigen::NumericalIssue) {
        

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov_matrix);

        if (eigensolver.info() == Eigen::Success) {
            Eigen::MatrixXd eigenvalues = eigensolver.eigenvalues();
            Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();

            // Make eigenvalues positive to ensure a Cholesky-like decomposition
            eigenvalues = eigenvalues.cwiseMax(1e-12);

            // Reconstruct the Cholesky-like decomposition
            Eigen::MatrixXd cholcov_like = eigenvectors * eigenvalues.cwiseSqrt().asDiagonal() * eigenvectors.transpose();

            return cholcov_like;
        } else {
            std::cerr << "Eigenvalue decomposition also failed. Matrix may not be diagonalizable." << std::endl;
        }
    }

    return llt.matrixL();
}
