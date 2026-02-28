import numpy as np

dt = 1
wheel_base = 5
wheel_radius = 0.3

def f(state, u):
    # state = [x, y, z, dx, dy, dz, d2x, d2y, d2z, yaw, d_yaw, turn_angle, d_turn_angle]
    # u = [throttle, v, turn_angle]  
    output = state.copy()
    x, y, z = state[0], state[1], state[2]
    dx, dy, dz = state[3], state[4], state[5]
    d2x, d2y, d2z = state[6], state[7], state[8]
    yaw, d_yaw = state[9], state[10]
    turn_angle, d_turn_angle = state[11], state[12]
    throttle, v, turn_angle_u = u[0], u[1], u[2]

    output[12] = (turn_angle_u - turn_angle) / dt
    output[11] = turn_angle + output[12] * dt
    
    output[10] = (v * np.tan(output[11])) / wheel_base
    output[9] = yaw + output[10] * dt
    
    output[6] = d2x
    output[7] = d2y
    output[8] = d2z
    
    output[3] = v * np.cos(output[9])
    output[4] = v * np.sin(output[9]) 
    output[5] = dz + d2z * dt
    
    output[0] = x + output[3] * dt
    output[1] = y + output[4] * dt
    output[2] = z + output[5] * dt
    
    return output

def F_jacobian(state, u):
    dx, dy, dz = state[3], state[4], state[5]
    yaw = state[9]
    turn_angle = state[11]
    v = u[1]
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
    F[9, 3] = (dx / v) * (np.tan(turn_angle) / wheel_base) * dt
    F[9, 4] = (dy / v) * (np.tan(turn_angle) / wheel_base) * dt
    F[9, 11] = (v * dt) / (wheel_base * np.cos(turn_angle) ** 2)
    F[11, 12] = dt
    return F

def h(state):
    # state = [x, y, z, dx, dy, dz, d2x, d2y, d2z, yaw, d_yaw, turn_angle, d_turn_angle]
    # z = [x, y, rpm_fl, rpm_fr, rpm_rl, rpm_rr, d2x, d2y, d2z, yaw, d_yaw]
    x = state[0]
    y = state[1]
    yaw = state[9]
    d_yaw = state[10]
    dx, dy = state[3], state[4]
    v = np.sqrt(dx**2 + dy**2)
    rpm_expected = (v * 60) / (2 * np.pi * wheel_radius)
    rpm_fl = rpm_expected
    rpm_fr = rpm_expected
    rpm_rl = rpm_expected
    rpm_rr = rpm_expected
    d2x = state[6]
    d2y = state[7]
    d2z = state[8]
    return np.array([x, y, rpm_fl, rpm_fr, rpm_rl, rpm_rr, d2x, d2y, d2z, yaw, d_yaw])

def H_matrix(state):
    H = np.zeros((11, 13))    
    H[0, 0] = 1
    H[1, 1] = 1
    dx, dy = state[3], state[4]
    v = np.sqrt(dx**2 + dy**2) + 1e-6
    drpm_dv = 60 / (2 * np.pi * wheel_radius)
    dv_ddx = dx / v
    dv_ddy = dy / v
    for i in range(2, 6):
        H[i, 3] = drpm_dv * dv_ddx
        H[i, 4] = drpm_dv * dv_ddy
    H[6, 6] = 1
    H[7, 7] = 1
    H[8, 8] = 1
    H[9, 9] = 1
    H[10, 10] = 1
    return H