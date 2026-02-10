import numpy as np
import matplotlib.pyplot as plt
import math

dt = 1

# [x, y, z, dx, dy, dz, d²x, d²y, d²z, yaw, d_yaw, delta, d_delta]
Q = np.eye(13) * 0.01
R = np.eye(13) * 0.15
P_0 = np.eye(13) * 0.1
x_hat_0 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, np.pi / 2, 0, 0, 0])
x_hat_k = [x_hat_0]  # predictions array
P_k = [P_0]  # covariance array
x_hat_prev = x_hat_0
P_prev = P_0

L = 5  # wheelbase length in meters
turn_radius = 13  # turn radius for simulating true path

# h and f are non linear this time

def ackermann_model(input, dt, L):
    # x = [x, y, z, dx, dy, dz, d²x, d²y, d²z, yaw, d_yaw, delta, d_delta]
    output = input.copy()
    x = input[0]
    y = input[1]
    z = input[2]
    dx = input[3]
    dy = input[4]
    dz = input[5]
    d2x = input[6]
    d2y = input[7]
    d2z = input[8]
    yaw = input[9]
    d_yaw = input[10]
    delta = input[11]
    d_delta = input[12]
    v = np.sqrt(dx**2 + dy**2)
    omega = (v * np.tan(delta)) / L
    output[12] = d_delta
    output[11] = delta + output[12] * dt
    output[10] = (v * np.tan(output[11])) / L
    output[9] = yaw + output[10] * dt
    output[6] = d2x
    output[7] = d2y
    output[8] = d2z
    output[3] = v * np.cos(output[9])
    output[4] = v * np.sin(output[9])
    output[5] = dz + output[8] * dt
    output[0] = x + output[3] * dt
    output[1] = y + output[4] * dt
    output[2] = z + output[5] * dt
    return output

x_t = [x_hat_0]
x_t_prev = x_hat_0

for i in range(200):
    if i < 60:
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_t.append(x_t_prev)
    elif i < 100:
        target_delta = np.arctan(L / turn_radius)
        x_t_prev[11] = target_delta
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_t.append(x_t_prev)
    elif i < 160:
        x_t_prev[11] = 0
        x_t_prev[4] = -1
        x_t_prev[3] = 0
        x_t_prev[9] = -np.pi / 2
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_t.append(x_t_prev)
    else:
        target_delta = np.arctan(L / turn_radius)
        x_t_prev[11] = target_delta
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_t.append(x_t_prev)

z_k = []

for i in range(len(x_t)):
    noise = np.random.multivariate_normal(np.zeros_like(x_t[i]), R)
    z_k_curr = x_t[i] + noise
    z_k.append(z_k_curr)

def f_jacobian(state, dt, L):
    dx = state[3]
    dy = state[4]
    yaw = state[9]
    delta = state[11]
    v = np.sqrt(dx**2 + dy**2)
    F = np.eye(13)
    F[0, 3] = (dx / v) * np.cos(yaw) * dt
    F[0, 4] = (dy / v) * np.cos(yaw) * dt
    F[0, 9] = -v * np.sin(yaw) * dt
    F[1, 3] = (dx / v) * np.sin(yaw) * dt
    F[1, 4] = (dy / v) * np.sin(yaw) * dt
    F[1, 9] = v * np.cos(yaw) * dt
    F[2, 5] = dt
    F[2, 8] = 0.5 * dt**2
    F[3, 6] = dt
    F[4, 7] = dt
    F[5, 8] = dt
    F[9, 3] = (dx / v) * (np.tan(delta) / L) * dt
    F[9, 4] = (dy / v) * (np.tan(delta) / L) * dt
    F[9, 11] = (v * dt) / (L * np.cos(delta) ** 2)
    F[11, 12] = dt
    return F

for i in range(200):
    if i < 60:
        x_hat_curr_minus = ackermann_model(x_hat_prev, 1, L)
    elif i < 100:
        target_delta = np.arctan(L / turn_radius)
        x_hat_prev[11] = target_delta
        x_hat_curr_minus = ackermann_model(x_hat_prev, 1, L)
    elif i < 160:
        x_hat_prev[11] = 0
        x_hat_prev[4] = -1
        x_hat_prev[3] = 0
        x_hat_prev[9] = -np.pi / 2
        x_hat_curr_minus = ackermann_model(x_hat_prev, 1, L)
    else:
        target_delta = np.arctan(L / turn_radius)
        x_hat_prev[11] = target_delta
        x_hat_curr_minus = ackermann_model(x_hat_prev, 1, L)

    F = f_jacobian(x_hat_curr_minus, 1, L)
    P_minus = F @ P_prev @ F.T + Q

    z_curr = z_k[i]

    # NEED TO CHANGE IN REAL IMPLEMENTATION to h(x_hat_curr_minus)
    z_expected = x_hat_curr_minus

    H = np.eye(13)  # NEED TO CHANGE IN REAL IMPLEMENTATION

    residual = z_curr - z_expected
    residual[9] = (residual[9] + np.pi) % (2 * np.pi) - np.pi

    S = H @ P_minus @ H.T + R

    K = P_minus @ H.T @ np.linalg.inv(S)

    x_hat_new = x_hat_curr_minus + K @ residual

    P_new = (np.identity(13) - K @ H) @ P_minus @ (np.identity(13) - K @ H).T + K @ R @ K.T

    x_hat_prev = x_hat_new
    P_prev = P_new

    x_hat_k.append(x_hat_new)
    P_k.append(P_new)

x_t = np.array(x_t)
z_k = np.array(z_k)
x_hat_k = np.array(x_hat_k)

plt.plot(x_t[:, 0], x_t[:, 1], label="True Path")
plt.scatter(z_k[:, 0], z_k[:, 1], s=5, label="Sensor Measurements")
plt.plot(x_hat_k[:, 0], x_hat_k[:, 1], label="EKF Prediction")
plt.axis("equal")
plt.legend()
plt.show()