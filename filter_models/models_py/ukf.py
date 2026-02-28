import numpy as np

def ukf(alpha, beta, kappa, x_k, u_k, P_k, z_k, Q, R, f, h):
    lambdaa = alpha ** 2 * (x_k.size + kappa) - x_k.size
    gamma = np.sqrt(x_k.size + lambdaa)

    sigma_points = np.zeros((2 * x_k.size + 1, x_k.size))
    sqrt_P = np.linalg.cholesky(P_k)
    sigma_points[0] = x_k
    for i in range(x_k.size):
        sigma_points[i + 1] = x_k + gamma * sqrt_P[:, i]
        sigma_points[i + 1 + x_k.size] = x_k - gamma * sqrt_P[:, i]

    Y_i_k = np.zeros((2 * x_k.size + 1, x_k.size))
    for i in range(2 * x_k.size + 1):
        Y_i_k[i] = f(sigma_points[i], u_k)

    W_m = np.full(2 * x_k.size + 1, 0.5 / (x_k.size + lambdaa))
    W_m[0] = lambdaa / (x_k.size + lambdaa)
    W_c = np.full(2 * x_k.size + 1, 0.5 / (x_k.size + lambdaa))
    W_c[0] += (1 - alpha ** 2 + beta)

    x_bar_k = np.zeros(x_k.size)
    for i in range(2 * x_k.size + 1):
        x_bar_k += W_m[i] * Y_i_k[i]

    P_bar_k = np.zeros((x_k.size, x_k.size))
    for i in range(2 * x_k.size + 1):
        P_bar_k += W_c[i] * np.outer(Y_i_k[i] - x_bar_k, Y_i_k[i] - x_bar_k)
    P_bar_k += Q

    Z_i_k = np.zeros((2 * x_k.size + 1, z_k.size))
    for i in range(2 * x_k.size + 1):
        Z_i_k[i] = h(Y_i_k[i])

    mu_z = np.zeros(z_k.size)
    for i in range(2 * x_k.size + 1):
        mu_z += W_m[i] * Z_i_k[i]

    P_z = np.zeros((z_k.size, z_k.size))
    for i in range(2 * x_k.size + 1):
        P_z += W_c[i] * np.outer(Z_i_k[i] - mu_z, Z_i_k[i] - mu_z)
    P_z += R
    
    y_k = z_k - mu_z

    P_x_z = np.zeros((x_k.size, z_k.size))
    for i in range(2 * x_k.size + 1):
        P_x_z += W_c[i] * np.outer(Y_i_k[i] - x_bar_k, Z_i_k[i] - mu_z)
    
    K_k = P_x_z @ np.linalg.inv(P_z)
    x_new = x_bar_k + K_k @ y_k
    P_new = P_bar_k - K_k @ P_z @ K_k.T

    return x_new, P_new