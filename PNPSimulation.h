//
// Created by lancelot on 3/15/17.
//

#ifndef PNP_PNPSIMULATION_H
#define PNP_PNPSIMULATION_H

#include <boost/math/distributions.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>

#include <g2o/stuff/sampler.h>
#include <g2o/core/factory.h>

#include <g2o/stuff/sampler.h>

#include <Eigen/Dense>
#include <sophus/se3.hpp>


template<typename T, int N>
void sampleGauss(Eigen::Matrix<T, N, 1>& mean,
                 Eigen::Matrix<T, N, N>& var,
                 std::vector<Eigen::Matrix<T, N, 1>>& vec, int num = 50) {
    g2o::GaussianSampler<Eigen::Matrix<T, N, 1>, Eigen::Matrix<T, N, N>> gaussSampler;
    gaussSampler.setDistribution(var);
    for (int i = 0; i < num; ++i) {
        Eigen::Matrix<T, N, 1> v = mean + gaussSampler.generateSample();
        vec.push_back(v);
    }
}

template <typename T, int N>
Eigen::Matrix<T, N, 1> oneSampleGauss(Eigen::Matrix<T, N, 1>& mean,
                 Eigen::Matrix<T, N, N>& var) {
    g2o::GaussianSampler<Eigen::Matrix<T, N, 1>, Eigen::Matrix<T, N, N>> gaussSampler;
    gaussSampler.setDistribution(var);
    return mean + gaussSampler.generateSample();
}


template <typename T, int N>
void sampleUniformMeans(T start, T end, std::vector<Eigen::Matrix<T, N, 1>>& means, int num = 50) {
    static boost::mt19937 rng(static_cast<unsigned>(std::time(0)));
    boost::uniform_real<T> uni_dist(start, end);
    Eigen::Matrix<T, N, 1> mean;
    for (int i = 0; i < num; ++i) {
        for(int dim = 0; dim < N; dim++)
            mean(dim) = uni_dist(rng);
        means.push_back(mean);
    }
}

template<typename T, int N>
void sampleGauss(std::vector<Eigen::Matrix<T, N, 1>>& means,
                 Eigen::Matrix<T, N, N>& var,
                 std::vector<Eigen::Matrix<T, N, 1>>& vec) {
    g2o::GaussianSampler<Eigen::Matrix<T, N, 1>, Eigen::Matrix<T, N, N>> gaussSampler;
    gaussSampler.setDistribution(var);
    for (size_t i = 0; i < means.size(); ++i) {
        Eigen::Matrix<T, N, 1> v = means[i] + gaussSampler.generateSample();
        vec.push_back(v);
    }
}

struct Camera {
    Camera(double fx, double fy, double cx, double cy) {
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
    }

    Eigen::Vector2d project(double x, double y, double z);
    Eigen::Vector2d project(Eigen::Vector3d point);
    Eigen::Vector3d bacProject(Eigen::Vector2d uv, double d);

    double fx_;
    double fy_;
    double cx_;
    double cy_;
};


class PNPSimulation {
public:
    PNPSimulation(Sophus::SE3d& se3, Eigen::Matrix<double, 2, 2>& Var);
    ~PNPSimulation();

    void start();

private:
    Sophus::SE3d real_;
    Eigen::Matrix2d information_;
};


#endif //PNP_PNPSIMULATION_H
