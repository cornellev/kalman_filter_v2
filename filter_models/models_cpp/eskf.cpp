#include <Eigen/Dense>
#include <functional>

std::pair<Eigen::VectorXd, Eigen::MatrixXd> eskf(
    Eigen::VectorXd x_k,
    Eigen::VectorXd u_k,
    Eigen::MatrixXd P_k,
    Eigen::VectorXd z_k,
    Eigen::MatrixXd H,
    Eigen::MatrixXd Q,
    Eigen::MatrixXd R,
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f,
    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> F_jacobian, 
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> F_w_jacobian,
    std::function<Eigen::VectorXd(Eigen::VectorXd)> h) {

        Eigen::MatrixXd F_k = F_jacobian(x_k, u_k);
        Eigen::VectorXd x_nom = f(x_k, u_k);
        Eigen::MatrixXd F_w = F_w_jacobian(x_k, u_k);

        Eigen::MatrixXd P_minus = F_k * P_k * F_k.transpose() + F_w * Q * F_w.transpose();
        
        Eigen::VectorXd z_expected = h(x_nom);
        Eigen::VectorXd residual = z_k - z_expected;

        Eigen::MatrixXd S = H * P_minus * H.transpose() + R;
        Eigen::MatrixXd K = P_minus * H.transpose() * S.inverse();

        Eigen::VectorXd delta_x_hat = K * residual;

        Eigen::VectorXd x_nom_new = x_nom + delta_x_hat;
        Eigen::MatrixXd P_new = (Eigen::MatrixXd::Identity(P_k.rows(), P_k.cols()) - K * H) * P_minus * (Eigen::MatrixXd::Identity(P_k.rows(), P_k.cols()) - K * H).transpose() + K * R * K.transpose();
        
        return {x_nom_new, P_new};

}

