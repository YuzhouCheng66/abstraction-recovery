#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;

int main() {
    constexpr int B = 1000;
    constexpr int N = 6;
    constexpr int R = 100;          // repeat 100 times
    constexpr float eps = 1e-3f;

    using Mat6f = Eigen::Matrix<float, N, N>;
    using Vec6f = Eigen::Matrix<float, N, 1>;

    // ----------------------------
    // 预构造（不计时）：A = M M^T + eps I
    // ----------------------------
    std::vector<Mat6f> A_batch(B);
    for (int i = 0; i < B; ++i) {
        Mat6f M = Mat6f::Random();                 // [-1,1] uniform
        Mat6f A = M * M.transpose();
        A.diagonal().array() += eps;
        A_batch[i] = A;
    }

    std::vector<Mat6f> inv_batch(B);
    const Mat6f I = Mat6f::Identity();

    // ----------------------------
    // benchmark：只计 inv 计算
    // ----------------------------
    std::vector<double> times_ms;
    times_ms.reserve(R);

    for (int r = 0; r < R; ++r) {
        auto t0 = Clock::now();

        #pragma omp parallel for
        for (int i = 0; i < B; ++i) {
            // 方式1：显式 inverse（通用）
            // inv_batch[i] = A_batch[i].inverse();

            // 方式2：SPD 专用等价 inv：LLT 分解 + solve(I)
            // 这在数值上与 inv 等价，但更符合“SPD 求逆”的工程写法
            Eigen::LLT<Mat6f> llt(A_batch[i]);
            inv_batch[i] = llt.solve(I);
        }

        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // 防止整体被优化掉：取一个值做 sink
    volatile float sink = inv_batch[0](0, 0);
    (void)sink;

    double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();

    std::cout << "Eigen serial inv benchmark (compute-only)\n";
    std::cout << "B=" << B << ", N=" << N << ", repeats=" << R << "\n";
    std::cout << "Average time per repeat: " << avg_ms << " ms\n";
    std::cout << "Average per 6x6 inverse: " << (avg_ms * 1000.0 / B) << " us\n";

    return 0;
}
