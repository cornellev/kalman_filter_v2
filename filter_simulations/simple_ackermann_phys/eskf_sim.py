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

x_truth = [x_hat_0]
x_t_prev = x_hat_0

for i in range(200):
    u = sim.get_control(i, L, turn_radius)
    x_t_prev = sim.ackermann_model(x_t_prev, u, dt, L)
    x_truth.append(x_t_prev)

# imu = [d2x, d2y, d2z, yaw, d_yaw]
# gps = [x, y, z]
gps_k = []
imu_k = []

for i in x_truth:
    noise_imu = np.random.multivariate_normal(np.zeros(imu_dim), R_imu)
    noise_gps = np.random.multivariate_normal(np.zeros(gps_dim), R_gps)
    gps_k.append(i[0:3] + noise_gps)
    imu_k.append(i[6:11] + noise_imu)

z_k = []
for i in range(len(gps_k)):
    z_k.append(sim.get_sensors(imu_k[i], gps_k[i]))
x_true_prev = x_hat_0
x_nom_prev = x_hat_0

for i in range(200):
    u = sim.get_control(i, L, turn_radius)

    F_k = sim.f_jacobian(x_nom_prev, u, dt, L)
    x_nom_prev = sim.ackermann_model(x_nom_prev, u, dt, L)

    F_w = np.eye(state_dim)

    P_minus = F_k @ P_prev @ F_k.T + F_w @ Q @ F_w.T

    z_pred = sim.h_state(x_nom_prev)
    residual = z_k[i] - z_pred
    residual[6] = (residual[6] + np.pi) % (2*np.pi) - np.pi

    S = H @ P_minus @ H.T + R
    K = P_minus @ H.T @ np.linalg.inv(S)

    delta_x_hat = K @ residual

    x_nom_prev = x_nom_prev + delta_x_hat

    P_prev = (np.eye(state_dim) - K @ H) @ P_minus @ (np.eye(state_dim) - K @ H).T + K @ R @ K.T
    x_hat_k.append(x_nom_prev)

    P_k.append(P_prev)

x_truth = np.array(x_truth)
z_k = np.array(z_k)
x_hat_k = np.array(x_hat_k)

sim.plot(x_truth, z_k, x_hat_k, "ESKF")