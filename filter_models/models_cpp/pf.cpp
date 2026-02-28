#include <Eigen/Dense>
#include <functional>
#include <random>
#include <numeric>

std::pair<Eigen::VectorXd, Eigen::MatrixXd> pf(
    Eigen::MatrixXd particles,
    Eigen::VectorXd x_k,
    Eigen::VectorXd u_k,
    Eigen::VectorXd z_k,
    Eigen::MatrixXd Q,
    Eigen::MatrixXd R,
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f,
    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> h) {

        std::random_device rd;
        std::mt19937 generator(rd());

        for (int i = 0; i < particles.rows(); ++i) {
            Eigen::MatrixXd L = Q.llt().matrixL();
            Eigen::VectorXd standard_normal(x_k.size());
            std::normal_distribution<double> dist(0.0, 1.0);
            for (int j = 0; j < x_k.size(); ++j) {
                standard_normal(j) = dist(generator);
            }
            Eigen::VectorXd noise = L * standard_normal;
            particles.row(i) = f(particles.row(i), u_k) + noise.transpose();
        }

        Eigen::VectorXd weights = Eigen::VectorXd::Zero(particles.rows());
        for (int i = 0; i < particles.rows(); ++i) {
            Eigen::VectorXd z_expected = h(particles.row(i), u_k);
            Eigen::VectorXd residual = z_k - z_expected;
            weights(i) = exp(-0.5 * residual.transpose() * R.inverse() * residual);
        }
        weights = weights / weights.sum();

        Eigen::VectorXd x_hat_k = Eigen::VectorXd::Zero(x_k.size());
        for (int i = 0; i < particles.rows(); ++i) {
            x_hat_k += weights(i) * particles.row(i).transpose();
        }

        double N_eff = 1 / (weights.array().square().sum());
        if (N_eff < particles.rows() / 2) {
            std::discrete_distribution<int> resample_dist(weights.data(), weights.data() + weights.size());
            Eigen::MatrixXd particles_new = Eigen::MatrixXd::Zero(particles.rows(), particles.cols());
            for (int i = 0; i < particles.rows(); ++i) {
                int idx = resample_dist(generator);
                particles_new.row(i) = particles.row(idx);
            }
            particles = particles_new;
        }

        return {x_hat_k, particles};

}