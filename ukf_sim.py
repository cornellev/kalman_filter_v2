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
n = 13
P_k = [P_0]  # covariance array
x_hat_prev = x_hat_0
P_prev = P_0

L = 5  # wheelbase length in meters
turn_radius = 13  # turn radius for simulating true path

alpha = 0.001
beta = 2
kappa = 0

lambdaa = alpha ** 2 * (n + kappa) - n
gamma = math.sqrt(n + lambdaa)

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

x_truth = [x_hat_0]
x_t_prev = x_hat_0

for i in range(200):
    if i < 60:
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_truth.append(x_t_prev)
    elif i < 100:
        target_delta = np.arctan(L / turn_radius)
        x_t_prev[11] = target_delta
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_truth.append(x_t_prev)
    elif i < 160:
        x_t_prev[11] = 0
        x_t_prev[4] = -1
        x_t_prev[3] = 0
        x_t_prev[9] = -np.pi / 2
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_truth.append(x_t_prev)
    else:
        target_delta = np.arctan(L / turn_radius)
        x_t_prev[11] = target_delta
        x_t_prev = ackermann_model(x_t_prev, 1, L)
        x_truth.append(x_t_prev)

z_k = []
m = 13

for i in range(len(x_truth)):
    noise = np.random.multivariate_normal(np.zeros_like(x_truth[i]), R)
    z_k_curr = x_truth[i] + noise
    z_k.append(z_k_curr)

for i in range(200):
    sqrt_P = np.linalg.cholesky(P_prev)

    sigma_points = []
    sigma_points.append(x_hat_prev)

    for j in range(n):
        sigma_points.append(x_hat_prev + gamma * sqrt_P[:, j])
        sigma_points.append(x_hat_prev - gamma * sqrt_P[:, j])

    Y_i_t = []

    for j in range(2 * n + 1):
        Y_i_t.append(ackermann_model(sigma_points[j], dt, L))
    
    W_m = np.full(2 * n + 1, 1 / (2 * (n + lambdaa)))
    W_m[0] = lambdaa / (n + lambdaa)
    W_c = np.full(2 * n + 1, 1 / (2 * (n + lambdaa)))
    W_c[0] = lambdaa / (n + lambdaa) + (1 - alpha ** 2 + beta)

    x_bar_t = np.zeros(n)
    
    for j in range(2 * n + 1):
        x_bar_t += Y_i_t[j] * W_m[j]

    P_bar_t = np.zeros((n, n))

    for j in range(2 * n + 1):
        P_bar_t += W_c[j] * np.outer(Y_i_t[j] - x_bar_t, Y_i_t[j] - x_bar_t)
        
    P_bar_t += Q

    z_t = z_k[i]

    # NEED TO CHANGE IN REAL IMPLEMENTATION to Z_i_t = h(Y_i_t)
    Z_i_t = Y_i_t

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

x_truth= np.array(x_truth)
z_k = np.array(z_k)
x_hat_k = np.array(x_hat_k)

plt.plot(x_truth[:, 0], x_truth[:, 1], label="True Path")
plt.scatter(z_k[:, 0], z_k[:, 1], s=5, label="Sensor Measurements")
plt.plot(x_hat_k[:, 0], x_hat_k[:, 1], label="UKF Prediction")
plt.axis("equal")
plt.legend()
plt.show()