#pragma once
#include <cstdint>
#include <manif/SE2.h>
#include <random>

constexpr double M_PI = 3.141592653589793238462643383279502884;

class RGen {
  std::mt19937 gen_sensor_, gen_motion_, gen_sample_;
  std::uniform_real_distribution<double> u_real_;
  std::normal_distribution<double> n_real_;

public:
  enum class Type { SENSOR, MOTION, SAMPLE };
  static RGen &instance() {
    static RGen instance_;
    return instance_;
  }
  static void seed(uint32_t seed) {
    RGen::instance().gen_sensor_.seed(seed);
    RGen::instance().gen_motion_.seed(seed + 1);
    RGen::instance().gen_sample_.seed(seed + 2);
  }

  static std::mt19937 &gen(Type type) {
    switch (type) {
    case Type::SENSOR:
      return RGen::instance().gen_sensor_;
    case Type::MOTION:
      return RGen::instance().gen_motion_;
    case Type::SAMPLE:
      return RGen::instance().gen_sample_;
    default:
      return RGen::instance().gen_sensor_;
    }
  }
  static double uniform(Type type = Type::SENSOR) { return RGen::instance().u_real_(gen(type)); }
  static double uniform01(Type type = Type::SENSOR) { return (RGen::uniform(type) + 1.f) / 2; }
  static int uniform_int(int min, int max, Type type = Type::SENSOR) {
    return std::uniform_int_distribution<int>(min, max)(gen(type));
  }
  static double normal(Type type = Type::SENSOR) { return RGen::instance().n_real_(gen(type)); }
  static Eigen::VectorXd normal_noise(const Eigen::VectorXd &std, Type type = Type::SENSOR) {
    Eigen::VectorXd noise(std.size());
    for (int32_t i = 0; i < noise.size(); ++i) {
      noise(i) = RGen::normal(type) * std(i);
    }
    return noise;
  }

  static Eigen::VectorXd add_normal_noise(const Eigen::VectorXd &m, const Eigen::VectorXd &std,
                                          Type type = Type::SENSOR) {
    return m + RGen::normal_noise(std, type);
  }

  static manif::SE2d pose() {
    return manif::SE2d(RGen::uniform(), RGen::uniform(), RGen::uniform() * M_PI);
  }
  static Eigen::Matrix3d precision() {
    Eigen::Matrix3d m;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        m(i, j) = RGen::uniform();
      }
    }
    Eigen::Matrix3d p = m.transpose() * m;
    Eigen::Matrix3d eps = Eigen::Vector3d(1e-3, 1e-3, 1e-3).asDiagonal();
    return p + eps;
  }

private:
  RGen() {
    u_real_ = std::uniform_real_distribution<>(-1, 1);
    n_real_ = std::normal_distribution<>(0, 1);
    gen_sensor_.seed(0);
  }

public:
  RGen(RGen const &) = delete;
  void operator=(RGen const &) = delete;
};
