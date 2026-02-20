import numpy as np
import matplotlib.pyplot as plt

dt = 1
state_dim = 13
imu_dim = 5
lidar_dim = 3

# x = [x, y, z, dx, dy, dz, d2x, d2y, d2z, yaw, d_yaw, delta, d_delta]
# u = [v, delta]
x_hat_0 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, np.pi / 2, 0, 0, 0])
u_0 = [1, 0]

R_imu = np.eye(imu_dim) * 0.15
R_lidar = np.eye(lidar_dim) * 0.15

L = 5  # wheelbase length in meters
turn_radius = 40  # turn radius for simulating true path

# h and f are non linear this time

def ackermann_model(state, u, dt, L):
    # x = [x, y, z, dx, dy, dz, d2x, d2y, d2z, yaw, d_yaw, delta, d_delta]
    # u = [v, delta]
    output = state.copy()
    x, y, z = state[0], state[1], state[2]
    dx, dy, dz = state[3], state[4], state[5]
    d2x, d2y, d2z = state[6], state[7], state[8]
    yaw, d_yaw = state[9], state[10]
    d_delta = state[12]
    v, delta = u[0], u[1]
    omega = (v * np.tan(delta)) / L
    output[12] = d_delta
    output[11] = delta
    output[10] = (v * np.tan(delta)) / L
    output[9] = yaw + output[10] * dt
    output[8] = d2z
    output[7] = d2y
    output[6] = d2x
    output[5] = dz + output[8] * dt
    output[4] = v * np.sin(output[9])
    output[3] = v * np.cos(output[9])
    output[2] = z + output[5] * dt
    output[1] = y + output[4] * dt
    output[0] = x + output[3] * dt
    return output


def f_jacobian(state, u, dt, L):
    dx = state[3]
    dy = state[4]
    yaw = state[9]
    delta = u[1]
    v = u[0]
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


def get_control(i, L, turn_radius):
    target_delta = np.arctan(L * np.pi / 40)
    v = 1
    if i < 60:
        delta = 0
    elif i < 100:
        delta = target_delta
    elif i < 160:
        delta = 0
    else:
        delta = target_delta
    return np.array([v, delta])


def get_sensors(imu, lidar):
    return np.concatenate((lidar, imu))

def h_state(state):
    return np.array([
        state[0],
        state[1],
        state[2],
        state[6],
        state[7],
        state[8],
        state[9],
        state[10],
    ])

def H_matrix():
    H = np.zeros((8, 13))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    H[3, 6] = 1
    H[4, 7] = 1
    H[5, 8] = 1
    H[6, 9] = 1
    H[7, 10] = 1
    return H

def R_matrix():
    return np.block([
        [R_lidar, np.zeros((lidar_dim, imu_dim))],
        [np.zeros((imu_dim, lidar_dim)), R_imu]
    ])

def plot(x_t, z_k, x_hat_k, method):
    fig, axes = plt.subplots(4, 4, figsize=(18, 12))
    axes = axes.flatten()

    state_labels = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'd2x', 'd2y', 'd2z', 'yaw', 'd_yaw', 'delta', 'd_delta']

    for i, label in enumerate(state_labels):
        ax = axes[i]
        ax.plot(x_t[:, i], label='True', linewidth=1.5)
        ax.plot(x_hat_k[:, i], label=method, linewidth=1, linestyle='--')
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True)

    measured_state_indices = {0: 0, 1: 1, 2: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7}

    for state_idx, z_idx in measured_state_indices.items():
        axes[state_idx].scatter(range(len(z_k)), z_k[:, z_idx], s=2, label='Measured', color='orange', zorder=3)
        axes[state_idx].legend(fontsize=7)

    ax_xy = axes[13]
    ax_xy.plot(x_t[:, 0], x_t[:, 1], label='True Path')
    ax_xy.scatter(z_k[:, 0], z_k[:, 1], s=5, label='Sensor Measurements')
    ax_xy.plot(x_hat_k[:, 0], x_hat_k[:, 1], label=method + " Prediction", linestyle='--')
    ax_xy.set_title('xy path')
    ax_xy.axis('equal')
    ax_xy.legend(fontsize=7)
    ax_xy.grid(True)

    axes[14].set_visible(False)
    axes[15].set_visible(False)

    plt.tight_layout()
    plt.show()