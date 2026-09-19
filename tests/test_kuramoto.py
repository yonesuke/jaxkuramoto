import jax.numpy as jnp
import pytest

from jaxkuramoto import Kuramoto, SakaguchiKuramoto


def test_kuramoto_initialization():
    omegas = jnp.array([1.0, 2.0, 3.0])
    model = Kuramoto(omegas=omegas, K=1.5)
    assert model.K == 1.5
    assert model.n_oscillator == 3


def test_kuramoto_vector_fn_synchronized():
    # When all phases are identical, the interaction term is zero, so dtheta/dt = omega
    omegas = jnp.array([1.0, 2.0, 3.0])
    model = Kuramoto(omegas=omegas, K=2.0)
    thetas = jnp.array([0.5, 0.5, 0.5])
    dthetas = model.vector_fn(0.0, thetas)
    assert jnp.allclose(dthetas, omegas, atol=1e-5)


def test_kuramoto_orderparameter():
    omegas = jnp.array([1.0, 2.0])
    model = Kuramoto(omegas=omegas, K=1.0)
    # Fully synchronized (same phases) -> order parameter = 1.0
    thetas_sync = jnp.array([1.0, 1.0])
    r_sync = model.orderparameter(0.0, thetas_sync)
    assert jnp.isclose(r_sync, 1.0, atol=1e-5)

    # Inverted phases (pi apart) -> order parameter = 0.0
    thetas_opp = jnp.array([0.0, jnp.pi])
    r_opp = model.orderparameter(0.0, thetas_opp)
    assert jnp.isclose(r_opp, 0.0, atol=1e-5)


def test_sakaguchi_kuramoto():
    omegas = jnp.array([1.0, 2.0, 3.0])
    alpha = 0.2
    model = SakaguchiKuramoto(omegas=omegas, K=1.0, alpha=alpha)
    assert model.alpha == alpha
    thetas = jnp.array([0.0, 0.5, 1.0])
    dthetas = model.vector_fn(0.0, thetas)
    assert dthetas.shape == (3,)
    r = model.orderparameter(0.0, thetas)
    assert 0.0 <= r <= 1.0
