#pragma once
#include <Eigen/Dense>

// Solves a linear system Ax = b
Eigen::VectorXd solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);