import jax
import jax.numpy as jnp

def ekf(x_k, u_k, P_k, z_k, Q, R, f, F_jacobian, H_jacobian, h, Re, L, C, dt):
    x_hat = f(x_k, u_k, Re, L, C, dt)
    F = F_jacobian(x_k, u_k, Re, L, C, dt)
    P_minus = F @ P_k @ F.T + Q

    z_expected = h(x_hat, Re)
    residual = z_k - z_expected

    H_k = H_jacobian(x_hat, Re)
    S = H_k @ P_minus @ H_k.T + R
    K = P_minus @ H_k.T @ jnp.linalg.inv(S)

    NIS = (residual @ jnp.linalg.inv(S) @ residual).squeeze()   # scalar

    x_new = x_hat + K @ residual
    P_new = (jnp.eye(P_k.shape[0]) - K @ H_k) @ P_minus @ (jnp.eye(P_k.shape[0]) - K @ H_k).T + K @ R @ K.T
    
    return x_new, P_new, residual, S, NIS