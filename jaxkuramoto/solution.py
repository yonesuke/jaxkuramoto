from dataclasses import dataclass

import jax.numpy as jnp

@dataclass
class Solution:
    """Solution class.

    Attributes:
        t0 (float): Initial time.
        t1 (float): Final time.
        ts (jnp.ndarray): Time array.
        init_state (jnp.ndarray): Initial state.
        final_state (jnp.ndarray): Final state.
        observables (jnp.ndarray): Observables.
    """
    t0: float
    t1: float
    ts: jnp.ndarray
    init_state: jnp.ndarray
    final_state: jnp.ndarray
    observables: jnp.ndarray

    def __repr__(self) -> str:
        return f"""Solution(
            t0 = {self.t0}, t1 = {self.t1},
            ts = {self.ts},
            init_state  = {self.init_state},
            final_state = {self.final_state},
            observables = {self.observables}
        )"""