import numpy as np
import track_sim_sensors as sim
import math

dt = 1
n = 13
imu_dim = 5
lidar_dim = 3
m = 8

# x = [x, y, z, dx, dy, dz, d2x, d2y, d2z, yaw, d_yaw, delta, d_delta]
# u = [v, delta]
x_hat_0 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, np.pi / 2, 0, 0, 0])
u_0 = [1, 0]
P_0 = np.eye(n) * 0.1
x_hat_k = [x_hat_0]  # predictions array
P_k = [P_0]  # covariance array
x_hat_prev = x_hat_0
P_prev = P_0

Q = np.eye(n) * 0.01
R_imu = np.eye(imu_dim) * 0.15
R_lidar = np.eye(lidar_dim) * 0.15
R = sim.R_matrix()

L = 5  # wheelbase length in meters
turn_radius = 40  # turn radius for simulating true path

x_truth = [x_hat_0]
x_t_prev = x_hat_0

alpha = 0.001
beta = 2
kappa = 0

lambdaa = alpha**2 * (n + kappa) - n
gamma = math.sqrt(n + lambdaa)

for i in range(200):
    u = sim.get_control(i, L, turn_radius)
    x_t_prev = sim.ackermann_model(x_t_prev, u, dt, L)
    x_truth.append(x_t_prev)

# imu = [d2x, d2y, d2z, yaw, d_yaw]
# lidar = [x, y, z]
lidar_k = []
imu_k = []

for i in x_truth:
    noise_imu = np.random.multivariate_normal(np.zeros(imu_dim), R_imu)
    noise_lidar = np.random.multivariate_normal(np.zeros(lidar_dim), R_lidar)
    lidar_k.append(i[0:3] + noise_lidar)
    imu_k.append(i[6:11] + noise_imu)

z_k = []
for i in range(len(lidar_k)):
    z_k.append(sim.get_sensors(imu_k[i], lidar_k[i]))

for i in range(200):
    u = sim.get_control(i, L, turn_radius)
    sqrt_P = np.linalg.cholesky(P_prev)

    sigma_points = []
    sigma_points.append(x_hat_prev)

    for j in range(n):
        sigma_points.append(x_hat_prev + gamma * sqrt_P[:, j])
        sigma_points.append(x_hat_prev - gamma * sqrt_P[:, j])

    Y_i_t = []

    for j in range(2 * n + 1):
        Y_i_t.append(sim.ackermann_model(sigma_points[j], u, dt, L))

    W_m = np.full(2 * n + 1, 1 / (2 * (n + lambdaa)))
    W_m[0] = lambdaa / (n + lambdaa)
    W_c = np.full(2 * n + 1, 1 / (2 * (n + lambdaa)))
    W_c[0] = lambdaa / (n + lambdaa) + (1 - alpha**2 + beta)

    x_bar_t = np.zeros(n)

    for j in range(2 * n + 1):
        x_bar_t += Y_i_t[j] * W_m[j]

    P_bar_t = np.zeros((n, n))

    for j in range(2 * n + 1):
        P_bar_t += W_c[j] * np.outer(Y_i_t[j] - x_bar_t, Y_i_t[j] - x_bar_t)

    P_bar_t += Q

    z_t = z_k[i]

    Z_i_t = [sim.h_state(y) for y in Y_i_t]

    mu_z = np.zeros(m)

    for j in range(2 * n + 1):
        mu_z += Z_i_t[j] * W_m[j]

    P_z = np.zeros((m, m))

    for j in range(2 * n + 1):
        P_z += W_c[j] * np.outer(Z_i_t[j] - mu_z, Z_i_t[j] - mu_z)

    P_z += R

    y_t = z_t - mu_z

    P_x_z = np.zeros((n, m))

    for j in range(2 * n + 1):
        P_x_z += W_c[j] * np.outer(Y_i_t[j] - x_bar_t, Z_i_t[j] - mu_z)

    K_t = P_x_z @ np.linalg.solve(P_z, np.eye(m))

    x_t = x_bar_t + K_t @ y_t
    P_t = P_bar_t - K_t @ P_z @ K_t.T

    x_hat_prev = x_t
    P_prev = P_t

    x_hat_k.append(x_t)
    P_k.append(P_t)

x_truth = np.array(x_truth)
z_k = np.array(z_k)
x_hat_k = np.array(x_hat_k)

sim.plot(x_truth, z_k, x_hat_k, "UKF")