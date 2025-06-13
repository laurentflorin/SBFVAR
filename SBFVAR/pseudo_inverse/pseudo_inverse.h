#pragma once
#include <Eigen/Dense>

// Computes the Moore-Penrose pseudo-inverse of a matrix
Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& mat);