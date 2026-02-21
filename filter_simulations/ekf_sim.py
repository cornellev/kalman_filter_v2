import numpy as np
import track_sim_sensors as sim

dt = 1
state_dim = 13
imu_dim = 5
gps_dim = 3

# x = [x, y, z, dx, dy, dz, d2x, d2y, d2z, yaw, d_yaw, delta, d_delta]
# u = [v, delta]
x_hat_0 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, np.pi / 2, 0, 0, 0])
u_0 = [1, 0]
P_0 = np.eye(state_dim) * 0.1
x_hat_k = [x_hat_0]  # predictions array
P_k = [P_0]  # covariance array
x_hat_prev = x_hat_0
P_prev = P_0

Q = np.eye(state_dim) * 0.01
R_imu = np.eye(imu_dim) * 0.15
R_gps = np.eye(gps_dim) * 0.15
R = sim.R_matrix()
H = sim.H_matrix()

L = 5  # wheelbase length in meters
turn_radius = 40  # turn radius for simulating true path

x_t = [x_hat_0]
x_t_prev = x_hat_0

for i in range(200):
    u = sim.get_control(i, L, turn_radius)
    x_t_prev = sim.ackermann_model(x_t_prev, u, dt, L)
    x_t.append(x_t_prev)

# imu = [d2x, d2y, d2z, yaw, d_yaw]
# gps = [x, y, z]
gps_k = []
imu_k = []

for i in x_t:
    noise_imu = np.random.multivariate_normal(np.zeros(imu_dim), R_imu)
    noise_gps = np.random.multivariate_normal(np.zeros(gps_dim), R_gps)
    gps_k.append(i[0:3] + noise_gps)
    imu_k.append(i[6:11] + noise_imu)

z_k = []
for i in range(len(gps_k)):
    z_k.append(sim.get_sensors(imu_k[i], gps_k[i]))


for i in range(200):
    u = sim.get_control(i, L, turn_radius)

    x_hat_curr_minus = sim.ackermann_model(x_hat_prev, u, dt, L)
    F = sim.f_jacobian(x_hat_prev, u, dt, L)
    P_minus = F @ P_prev @ F.T + Q

    z_curr = z_k[i]

    z_expected = sim.h_state(x_hat_curr_minus)

    residual = z_curr - z_expected
    residual[6] = (residual[6] + np.pi) % (2 * np.pi) - np.pi

    S = H @ P_minus @ H.T + R
    K = P_minus @ H.T @ np.linalg.inv(S)

    x_hat_new = x_hat_curr_minus + K @ residual
    P_new = (np.eye(state_dim) - K @ H) @ P_minus @ (np.eye(state_dim) - K @ H).T + K @ R @ K.T

    x_hat_prev = x_hat_new
    P_prev = P_new
    x_hat_k.append(x_hat_new)
    P_k.append(P_new)

x_t = np.array(x_t)
z_k = np.array(z_k)
x_hat_k = np.array(x_hat_k)

sim.plot(x_t, z_k, x_hat_k, "EKF")