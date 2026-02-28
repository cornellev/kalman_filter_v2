import numpy as np

def eskf(x_k, u_k, P_k, z_k, H, Q, R, f, F_jacobian, F_w_jacobian, h):
    F_k = F_jacobian(x_k, u_k)
    x_nom = f(x_k, u_k)
    F_w = F_w_jacobian(x_k, u_k)

    P_minus = F_k @ P_k @ F_k.T + F_w @ Q @ F_w.T
    
    z_expected = h(x_nom)
    residual = z_k - z_expected

    S = H @ P_minus @ H.T + R
    K = P_minus @ H.T @ np.linalg.inv(S)

    delta_x_hat = K @ residual

    x_nom_new = x_nom + delta_x_hat
    P_new = (np.eye(P_k.shape[0]) - K @ H) @ P_minus @ (np.eye(P_k.shape[0]) - K @ H).T + K @ R @ K.T
    
    return x_nom_new, P_new