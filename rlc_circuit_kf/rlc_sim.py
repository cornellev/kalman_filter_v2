import jax
import jax.numpy as jnp
import ekf
import rlc_physics as sim
from scipy.stats import chi2

Re = 1
C = 0.5
L = 0.1
dt = 0.1


state_dim = 2
sensor_dim = 1

lam0 = chi2.ppf(0.95, df=sensor_dim)

state = jnp.array([3.3, 0.0])
P = jnp.eye(state_dim) * 0.1
Q = jnp.eye(state_dim) * 0.01
R = jnp.eye(sensor_dim) * 0.15

F_jac = jax.jacfwd(sim.f)
H_jac = jax.jacfwd(sim.h)

for _ in range(500):
    u = jnp.array([1.0])
    z = jnp.array([0.0])
    state, P, residual, S, NIS = ekf.ekf(state, u, P, z, Q, R, sim.f, F_jac, H_jac, sim.h, Re, L, C, dt)
    accepted = NIS <= lam0
    print(f"state: {state}  NIS: {NIS:.3f}  {'OK' if accepted else 'REJECTED'}")