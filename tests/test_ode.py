import jax.numpy as jnp
import pytest

from jaxkuramoto import Kuramoto, odeint
from jaxkuramoto.solver import euler, runge_kutta


def test_odeint_with_euler_and_runge_kutta():
    omegas = jnp.array([1.0, 2.0, 3.0])
    model = Kuramoto(omegas=omegas, K=1.0)
    init_state = jnp.zeros(3)

    # Euler solver
    sol_euler = odeint(model.vector_fn, euler, 0.0, 1.0, 0.1, init_state)
    assert sol_euler.final_state.shape == (3,)
    assert sol_euler.observables.shape == (10, 3)

    # Runge-Kutta solver
    sol_rk = odeint(model.vector_fn, runge_kutta, 0.0, 1.0, 0.1, init_state)
    assert sol_rk.final_state.shape == (3,)
    assert sol_rk.observables.shape == (10, 3)


def test_odeint_with_observable_fn():
    omegas = jnp.array([1.0, 2.0, 3.0])
    model = Kuramoto(omegas=omegas, K=1.0)
    init_state = jnp.zeros(3)

    sol = odeint(
        model.vector_fn,
        runge_kutta,
        0.0,
        1.0,
        0.1,
        init_state,
        observable_fn=model.orderparameter,
    )
    assert sol.observables.shape == (10,)
    assert jnp.all((sol.observables >= 0.0) & (sol.observables <= 1.0 + 1e-5))


def test_odeint_invalid_inputs():
    model = Kuramoto(omegas=jnp.array([1.0]), K=1.0)
    init_state = jnp.zeros(1)

    # t0 >= t1
    with pytest.raises(ValueError, match="t0 must be smaller than t1"):
        odeint(model.vector_fn, euler, 1.0, 0.0, 0.1, init_state)

    # dt <= 0
    with pytest.raises(ValueError, match="dt must be positive"):
        odeint(model.vector_fn, euler, 0.0, 1.0, -0.1, init_state)
