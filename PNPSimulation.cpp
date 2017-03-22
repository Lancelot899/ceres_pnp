//
// Created by lancelot on 3/15/17.
//

#include <memory>

#include "PNPSimulation.h"

#include <ceres/ceres.h>

Eigen::Vector2d Camera::project(double x, double y, double z) {
    Eigen::Vector2d uv;
    uv(0) = fx_ * x / z + cx_;
    uv(1) = fy_ * y / z + cy_;

    return uv;
}

Eigen::Vector2d Camera::project(Eigen::Vector3d point) {
    return project(point(0), point(1), point(2));
}

Eigen::Vector3d Camera::bacProject(Eigen::Vector2d uv, double d) {
    Eigen::Vector3d point;
    point(2) = d;
    point(0) = (uv(0) - cx_) * d / fx_;
    point(1) = (uv(1) - cy_) * d / fy_;

    return point;
}

/* ############################################################################################
 * ############################################################################################
 */

class reprojectErr : public ceres::SizedCostFunction<2, 6> {
public:
    reprojectErr(Eigen::Vector3d& pt, Eigen::Vector2d &uv,
                 Eigen::Matrix<double, 2, 2> &information,
                 std::shared_ptr<Camera> cam);
    virtual ~reprojectErr() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

public:
    Eigen::Vector3d pt_;
    Eigen::Vector2d uv_;
    std::shared_ptr<Camera> cam_;
    Eigen::Matrix<double, 2, 2> sqrt_information_;
    static int index;
};

int reprojectErr::index = 0;

bool reprojectErr::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(parameters[0]);
    Sophus::SE3d T = Sophus::SE3d::exp(lie);

    //std::cout << T.matrix3x4() << std::endl;

    Eigen::Vector3d P = T * pt_;
    Eigen::Vector2d uv = cam_->project(P);
    Eigen::Vector2d err = uv - uv_;
    err = sqrt_information_ * err;

    residuals[0] = err(0);
    residuals[1] = err(1);

    Eigen::Matrix<double, 2, 6> Jac = Eigen::Matrix<double, 2, 6>::Zero();
    Jac(0, 0) = cam_->fx_ / P(2); Jac(0, 2) = -P(0) / P(2) /P(2) * cam_->fx_; Jac(0, 3) = Jac(0, 2) * P(1);
    Jac(0, 4) = cam_->fx_ - Jac(0, 2) * P(0); Jac(0, 5) = -Jac(0, 0) * P(1);

    Jac(1, 1) = cam_->fy_ / P(2); Jac(1, 2) = -P(1) / P(2) /P(2) * cam_->fy_; Jac(1, 3) = -cam_->fy_ + Jac(1, 2) * P(1);
    Jac(1, 4) = -Jac(1, 2) * P(0); Jac(1, 5) = Jac(1, 1) * P(0);
    Jac = sqrt_information_ * Jac;

    int k = 0;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 6; ++j) {
            if(k >= 12)
                return false;
            if(jacobians) {
                if(jacobians[0])
                    jacobians[0][k] = Jac(i, j);
            }
            k++;
        }
    }

    //printf("jacobian ok!\n");

    return true;

}

reprojectErr::reprojectErr(Eigen::Vector3d& pt, Eigen::Vector2d &uv,
                           Eigen::Matrix<double, 2, 2>& information,
                           std::shared_ptr<Camera> cam) :   pt_(pt), uv_(uv), cam_(cam) {

    //printf("index = %d\n", index++);
    Eigen::LLT<Eigen::Matrix<double, 2, 2>> llt(information);
    sqrt_information_ = llt.matrixL();
}


class CERES_EXPORT SE3Parameterization : public ceres::LocalParameterization {
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

bool SE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const {
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    return true;
}

bool SE3Parameterization::Plus(const double* x,
                  const double* delta,
                  double* x_plus_delta) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

    Sophus::SE3d T = Sophus::SE3d::exp(lie);
    Sophus::SE3d delta_T = Sophus::SE3d::exp(delta_lie);
    Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

    for(int i = 0; i < 6; ++i) x_plus_delta[i] = x_plus_delta_lie(i, 0);

    return true;

}

/* ############################################################################################
 * ############################################################################################
 */


PNPSimulation::PNPSimulation(Sophus::SE3d &se3, Eigen::Matrix<double, 2, 2>& Var) {
    real_ = se3;
    information_ = Var.inverse();
}

void PNPSimulation::start() {
    std::vector<Eigen::Vector3d> ptReals;
    sampleUniformMeans<double, 3>(-20.0, 20.0, ptReals);

    std::shared_ptr<Camera> cam = std::make_shared<Camera>(1.0, 1.0, 1.0, 1.0);
    double se3[6] = {0, 0, 0, 0, 0, 0};
    std::vector<Eigen::Vector2d> turbulences;
    {
        Eigen::Matrix<double, 2, 1> turbulence_;
        Eigen::Matrix<double, 2, 2> var = information_.inverse();
        Eigen::Matrix<double, 2, 1> mean = Eigen::Matrix<double, 2, 1>::Zero();
        for(size_t i = 0; i < ptReals.size(); ++i) {
            turbulence_ = oneSampleGauss<double, 2>(mean, var);
            turbulences.push_back(turbulence_);
        }
    }

    std::vector<Eigen::Vector2d> pix;
    for(size_t i = 0; i < turbulences.size(); ++i) {
        pix.push_back(cam->project((real_ * ptReals[i])) + turbulences[i]);
    }

//    for(size_t i = 0; i < ptReal.size(); ++i) {
//        std::cout << "3d:\n" << ptReal[i] << "\n2d:\n" << pix[i] << "\n###########################" << std::endl;
//    }


    ceres::Problem problem;

    for(size_t i = 0; i < ptReals.size(); ++i) {
        ceres::CostFunction * costFun = new reprojectErr(ptReals[i], pix[i], information_, cam);
        problem.AddResidualBlock(costFun, new ceres::HuberLoss(0.5), se3);
        //
    }

    problem.SetParameterization(se3, new SE3Parameterization());

    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = true;
    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

}

PNPSimulation::~PNPSimulation() {}