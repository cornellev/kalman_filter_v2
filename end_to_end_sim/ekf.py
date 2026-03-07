import numpy as np

def ekf(x_k, u_k, P_k, z_k, Q, R, f, F_jacobian, H_jacobian, h):
    x_hat = f(x_k, u_k)
    F = F_jacobian(x_k, u_k)
    P_minus = F @ P_k @ F.T + Q

    z_expected = h(x_hat)
    residual = z_k - z_expected

    H_k = H_jacobian(x_hat)
    S = H_k @ P_minus @ H_k.T + R
    K = P_minus @ H_k.T @ np.linalg.inv(S)

    x_new = x_hat + K @ residual
    P_new = (np.eye(P_k.shape[0]) - K @ H_k) @ P_minus @ (np.eye(P_k.shape[0]) - K @ H_k).T + K @ R @ K.T
    
    return x_new, P_new