#include <iostream>
#include <chrono>
#include <Eigen/Dense>
int main() {
    const int N = 5000;
    std::cout << "Testing " << N << "x" << N << " matrix multiplication..." << std::endl;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd C = A * B;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "C(0,0) = " << C(0,0) << std::endl;
    return 0;
}
