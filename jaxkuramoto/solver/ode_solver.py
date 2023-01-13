from typing import Callable

import jax.numpy as jnp

VECTOR_FN = Callable[[float, jnp.ndarray], jnp.ndarray]

def euler(func: VECTOR_FN, t: float, dt: float, state: jnp.ndarray):
    """Euler method.
    """
    return state + dt * func(t, state)

def runge_kutta(func: VECTOR_FN, t: float, dt: float, state: jnp.ndarray):
    """Runge Kutta method.
    """
    k1 = func(t, state)
    k2 = func(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = func(t + dt, state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)