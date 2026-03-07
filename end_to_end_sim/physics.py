import numpy as np

dt = 1
wheel_base = 5
wheel_radius = 0.3

def f(state, u):
    # state = [x, y, dx, dy, d2x, d2y, turn_angle]
    # u = [throttle, v, turn_angle]
    output = state.copy()
    x, y = state[0], state[1]
    dx, dy = state[2], state[3]
    d2x, d2y = state[4], state[5]
    turn_angle = state[6]
    throttle, v, turn_angle_u = u[0], u[1], u[2]

    output[6] = turn_angle_u

    output[2] = v * np.cos(output[6])
    output[3] = v * np.sin(output[6])

    output[4] = (output[2] - dx) / dt
    output[5] = (output[3] - dy) / dt

    output[0] = x + output[2] * dt
    output[1] = y + output[3] * dt

    return output

def F_jacobian(state, u):
    dx, dy = state[2], state[3]
    d2x, d2y = state[4], state[5]
    throttle, v = u[0], u[1]
    F = np.eye(7)
    F[0, 3] = v * np.cos(dy) * dt
    F[1, 5] = throttle * np.cos(d2y) * dt
    F[2, 2] = -v * np.sin(dx)
    F[3, 3] = v * np.cos(dy)
    F[4, 4] = -throttle * np.sin(d2x)
    F[5, 5] = throttle * np.cos(d2y)
    F[6, 6] = 0.5
    return F


def h(state):
    # state = [x, y, dx, dy, d2x, d2y, turn_angle]
    # z = [x, y, rpm_fl, rpm_fr, rpm_rl, rpm_rr]
    x = state[0]
    y = state[1]
    dx = state[3]
    dy = state[4]
    v = np.sqrt(dx**2 + dy**2)
    rpm_expected = (v * 60) / (2 * np.pi * wheel_radius)
    rpm_fl = rpm_expected
    rpm_fr = rpm_expected
    rpm_rl = rpm_expected
    rpm_rr = rpm_expected
    return np.array([x, y, rpm_fl, rpm_fr, rpm_rl, rpm_rr])


def H_matrix(state):
    H = np.zeros((6, 7))
    H[0, 0] = 1
    H[1, 1] = 1
    dx, dy = state[2], state[3]
    v = np.sqrt(dx**2 + dy**2) + 1e-6
    drpm_dv = 60 / (2 * np.pi * wheel_radius)
    dv_ddx = dx / v
    dv_ddy = dy / v
    for i in range(2, 6):
        H[i, 2] = drpm_dv * dv_ddx
        H[i, 3] = drpm_dv * dv_ddy
    return H