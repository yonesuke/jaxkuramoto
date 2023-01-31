from typing import Callable

from jax.lax import fori_loop
from jax import jit
import jax.numpy as jnp

from .solution import Solution

VECTOR_FN = Callable[[float, jnp.ndarray], jnp.ndarray]
SOLVER = Callable[[VECTOR_FN, float, float, jnp.ndarray], jnp.ndarray]
OBSERVABLE_FN = Callable[[jnp.ndarray], jnp.ndarray]

class ODE:
    """Ordinary Differential Equation (ODE) Class.
    """
    def __init__(self) -> None:
        pass

    def vector_fn(self, t: float, state: jnp.ndarray) -> jnp.ndarray:
        """Vector field of the ODE.
        """
        raise NotImplementedError()

def odeint(vector_fn: VECTOR_FN, solver: SOLVER, t0: float, t1: float, dt: float, init_state: jnp.ndarray, observable_fn: OBSERVABLE_FN=None) -> Solution:
    """Integrate the ODE.
    
    Args:
        vector_fn (VECTOR_FN): Vector field of the ODE.
        solver (SOLVER): Solver of the ODE.
        t0 (float): Initial time.
        t1 (float): Final time.
        dt (float): Time step.
        init_state (jnp.ndarray): Initial state.
        observable_fn (OBSERVABLE_FN, optional): Function to calculate observables. Defaults to None.
        
    Returns:
        Solution: Solution class.

    Raises:
        ValueError: If t0 >= t1 or dt <= 0.
    """
    # check input
    if t0 >= t1:
        raise ValueError("t0 must be smaller than t1.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    # create update function
    update_fn = jit(lambda t, state: solver(vector_fn, t, dt, state))
    ts = jnp.arange(t0, t1, dt)
    n_step = ts.shape[0]
    # check observable_fn
    if observable_fn is None:
        observable_fn = lambda _t, _x: _x
    # create observables list
    observable_1st = observable_fn(t0, init_state)
    observables = jnp.zeros(shape=(n_step, *(observable_1st.shape)), dtype=observable_1st.dtype)
    observables = observables.at[0].set(observable_1st)
    @jit
    def body_fn(i, val):
        state, _observables = val
        next_state = update_fn(ts[i], state)
        observable_i = observable_fn(ts[i], state)
        _observables = _observables.at[i].set(observable_i)
        return (next_state, _observables)
    # run!!
    final_state, observables = fori_loop(1, n_step, body_fn, (init_state, observables))
    # store the result to Solution class
    sol = Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        init_state=init_state,
        final_state=final_state,
        observables=observables
    )
    return sol

def odeint_resume():
    pass