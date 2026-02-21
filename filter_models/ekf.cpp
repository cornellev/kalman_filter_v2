#include <Eigen/Dense>
#include <functional>

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ekf(
    Eigen::VectorXd x_k,
    Eigen::VectorXd u_k,
    Eigen::MatrixXd P_k,
    Eigen::VectorXd z_k,
    Eigen::MatrixXd Q,
    Eigen::MatrixXd R,
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f,
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> F_jacobian, 
    std::function<Eigen::MatrixXd(Eigen::VectorXd)> H_jacobian,
    std::function<Eigen::VectorXd(Eigen::VectorXd)> h) {

        Eigen::VectorXd x_hat = f(x_k, u_k);
        Eigen::MatrixXd F = F_jacobian(x_k, u_k);
        Eigen::MatrixXd P_minus = F * P_k * F.transpose() + Q;

        Eigen::VectorXd z_expected = h(x_hat);
        Eigen::VectorXd residual = z_k - z_expected;

        Eigen::MatrixXd H_k = H_jacobian(x_hat);
        Eigen::MatrixXd S = H_k * P_minus * H_k.transpose() + R;
        Eigen::MatrixXd K = P_minus * H_k.transpose() * S.inverse();

        Eigen::VectorXd x_new = x_hat + K * residual;
        Eigen::MatrixXd P_new = (Eigen::MatrixXd::Identity(P_k.rows(), P_k.cols()) - K * H_k) * P_minus * (Eigen::MatrixXd::Identity(P_k.rows(), P_k.cols()) - K * H_k).transpose() + K * R * K.transpose();
        
        return {x_new, P_new};

}

