"""Model A dynamics: phi/x transforms and forward simulation via diffrax."""

import jax.numpy as jnp
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt


def phi_to_x(phi):
    x = jnp.log(phi)
    return x


def x_to_phi(x):
    phi = jnp.exp(x)
    return phi


def dynamics_func_x_constant(t, x, args):
    (chi, mu, clamp) = args
    N = chi.shape[0]
    d = jnp.ones(N)
    if clamp is not None:
        d = d.at[:clamp].set(0.0)

    phi = jnp.exp(x)
    mu_x = 1 + jnp.log(phi) + jnp.matmul(chi, phi)
    mu_diff = mu_x + mu

    # constant-mobility multiplier: weight = d
    lam = jnp.sum(d * mu_diff) / jnp.sum(d)

    # phi-space flux with constant mobility, then convert to x via 1/phi
    dphi = -d * (mu_diff - lam)
    per_x = dphi * jnp.exp(-x)
    return per_x


def dynamics_func_x_dphi(t, x, args):
    (chi, mu, clamp) = args

    N = chi.shape[0]
    d = jnp.ones(N)
    if clamp is not None:
        d = d.at[:clamp].set(0.0)

    phi = jnp.exp(x)
    mu_x = 1 + jnp.log(phi) + jnp.matmul(chi, phi)
    mu_diff = mu_x + mu
    d_phi = jnp.multiply(d, phi)
    sum_d_phi = jnp.sum(d_phi)
    per_x = -(jnp.multiply(d, mu_diff) - jnp.multiply(d, jnp.sum(jnp.multiply(d_phi, mu_diff) / sum_d_phi)))
    return per_x


def forward_sim_x_ssolvent_clamp(phi0,
                                  chi,
                                  mu,
                                  clamp,
                                  t_end,
                                  dt,
                                  max_steps,
                                  samples=10,
                                  mobility="dphi"):
    x0 = phi_to_x(phi0)
    if mobility == "dphi":
        term = ODETerm(dynamics_func_x_dphi)
    else:
        term = ODETerm(dynamics_func_x_constant)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.linspace(0, t_end, samples))

    sol = diffeqsolve(term,
                       solver,
                       t0=0,
                       t1=t_end,
                       dt0=dt,
                       y0=x0,
                       saveat=saveat,
                       args=(chi, mu, clamp),
                       max_steps=max_steps,
                       )

    return sol
