#include <Eigen/Dense>
#include <functional>

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ukf(
    double alpha,
    double beta,
    double kappa,
    Eigen::VectorXd x_k,
    Eigen::VectorXd u_k,
    Eigen::MatrixXd P_k,
    Eigen::VectorXd z_k,
    Eigen::MatrixXd Q,
    Eigen::MatrixXd R,
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f,
    std::function<Eigen::VectorXd(Eigen::VectorXd)> h) {
        
        double lambda = alpha * alpha * (x_k.size() + kappa) - x_k.size();
        double gamma = sqrt(x_k.size() + lambda);

        Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(2 * x_k.size() + 1, x_k.size());
        Eigen::MatrixXd L = P_k.llt().matrixL();
        sigma_points.row(0) = x_k;
        for (int i = 0; i < x_k.size(); ++i) {
            sigma_points.row(i + 1) = x_k + gamma * L.col(i);
            sigma_points.row(i + 1 + x_k.size()) = x_k - gamma * L.col(i);
        }

        Eigen::MatrixXd Y_i_k = Eigen::MatrixXd::Zero(2*x_k.size() + 1, x_k.size());

        for (int i = 0; i < 2 * x_k.size() + 1; ++i) {
            Y_i_k.row(i) = f(sigma_points.row(i), u_k);
        }

        Eigen::VectorXd W_m = Eigen::VectorXd::Constant(2*x_k.size() + 1, 0.5 / (x_k.size() + lambda));
        W_m(0) = lambda / (x_k.size() + lambda);
        Eigen::VectorXd W_c = W_m;
        W_c(0) = lambda / (x_k.size() + lambda) + (1 - alpha * alpha + beta);

        Eigen::VectorXd x_bar_k = Eigen::VectorXd::Zero(x_k.size());
        for (int i = 0; i < 2 * x_k.size() + 1; ++i)
            x_bar_k += W_m(i) * Y_i_k.row(i).transpose();

        Eigen::MatrixXd P_bar_k = Eigen::MatrixXd::Zero(x_k.size(), x_k.size());
        for (int i = 0; i < 2 * x_k.size() + 1; ++i) {
            Eigen::VectorXd y = Y_i_k.row(i).transpose() - x_bar_k;
            P_bar_k += W_c(i) * (y * y.transpose());
        }
        P_bar_k += Q;

        Eigen::MatrixXd Z_i_k(2 * x_k.size() + 1, z_k.size());
        for (int i = 0; i < 2 * x_k.size() + 1; ++i) {
            Z_i_k.row(i) = h(Y_i_k.row(i));
        }

        Eigen::VectorXd mu_z = Eigen::VectorXd::Zero(z_k.size());
        for (int i = 0; i < 2*x_k.size() + 1; ++i)
            mu_z += W_m(i) * Z_i_k.row(i).transpose();

        Eigen::MatrixXd P_z = Eigen::MatrixXd::Zero(z_k.size(), z_k.size());
        for (int i = 0; i < 2 * x_k.size() + 1; ++i)
            P_z += W_c(i) * (Z_i_k.row(i).transpose() - mu_z) * (Z_i_k.row(i).transpose() - mu_z).transpose();
        P_z += R;

        Eigen::VectorXd y_k = z_k - mu_z;

        Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(x_k.size(), z_k.size());
        for (int i = 0; i < 2 * x_k.size() + 1; ++i)
            P_xz += W_c(i) * (Y_i_k.row(i).transpose() - x_bar_k) * (Z_i_k.row(i).transpose() - mu_z).transpose();

        Eigen::MatrixXd K_k = P_xz * P_z.inverse();

        Eigen::VectorXd x_new = x_bar_k + K_k * y_k;
        Eigen::MatrixXd P_new = P_bar_k - K_k * P_z * K_k.transpose();

        return {x_new, P_new};
        
}