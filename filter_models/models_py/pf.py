import numpy as np

def pf(particles, x_k, u_k, z_k, Q, R, f, h):
    num_particles = particles.shape[0]
    for i in range(num_particles):
        noise = np.random.multivariate_normal(np.zeros(x_k.shape[0]), Q)
        particles[i] = f(particles[i], u_k) + noise

    weights = []
    for i in range(num_particles):
        z_expected = h(particles[i], u_k)
        residual = z_k - z_expected
        weights.append(np.exp(-0.5 * residual.T @ np.linalg.inv(R) @ residual))
    weights = np.array(weights)
    weights /= np.sum(weights)

    x_hat_k = np.sum(particles.T * weights, axis=1)

    N_eff = 1 / np.sum(weights ** 2)
    if N_eff < num_particles / 2:
        inds = np.random.choice(range(num_particles), size=num_particles, p=weights)
        particles = particles[inds]
        weights = np.ones(num_particles) / num_particles

    return x_hat_k, particles