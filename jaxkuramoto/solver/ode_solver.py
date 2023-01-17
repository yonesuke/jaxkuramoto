from typing import Callable

from jax.tree_util import tree_map
import jax.numpy as jnp

VECTOR_FN = Callable[[float, jnp.ndarray], jnp.ndarray]

_tmul = lambda a, x: tree_map(lambda _x: a * _x, x)
_tsum = lambda x, y: tree_map(lambda _x, _y: _x + _y, x, y)

def euler(func: VECTOR_FN, t: float, dt: float, state: jnp.ndarray):
    """Euler method.

    Args:
        func: A function of the form func(t, x) -> dx/dt.
        t: Current time.
        dt: Time step.
        state: Current state.

    Returns:
        Next state.
    """
    diff = _tmul(dt, func(t, state))
    return _tsum(state, diff)

def runge_kutta(func: VECTOR_FN, t: float, dt: float, state: jnp.ndarray):
    """Runge Kutta method.

    Args:
        func: A function of the form func(t, x) -> dx/dt.
        t: Current time.
        dt: Time step.
        state: Current state.

    Returns:
        Next state.
    """
    k1 = func(t, state)
    k2 = func(t + 0.5 * dt, _tsum(state, _tmul(0.5 * dt, k1)))
    k3 = func(t + 0.5 * dt, _tsum(state, _tmul(0.5 * dt, k2)))
    k4 = func(t + dt, _tsum(state, _tmul(dt, k3)))
    diff = tree_map(
        lambda _k1, _k2, _k3, _k4: (_k1 + 2.0 * _k2 + 2.0 * _k3 + _k4) * dt / 6.0,
        k1, k2, k3, k4
    )
    return _tsum(state, diff)