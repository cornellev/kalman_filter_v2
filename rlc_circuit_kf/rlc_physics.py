import jax
import jax.numpy as jnp

# x = [V_c, i_L]
# u = [i_s]
def f(x, u, R, L, C, dt):
    V_c, i_L = x
    i_s = u[0]
    dV_c_dt = (-1 * i_L + i_s) / C
    di_L_dt = (V_c - R * i_L) / L
    return jnp.array([V_c + dV_c_dt * dt, i_L + di_L_dt * dt])

def h(x, R):
    return jnp.array([R * x[1]])
