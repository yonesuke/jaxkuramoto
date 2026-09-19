import jax.numpy as jnp
import pytest

from jaxkuramoto import odeint
from jaxkuramoto.distribution import Cauchy, CauchyMultiply, Normal, Uniform
from jaxkuramoto.solver import runge_kutta
from jaxkuramoto.theory import OttAntonsen, critical_point, orderparam


def test_critical_point():
    c = Cauchy(loc=0.0, gamma=1.0)
    # For Cauchy distribution, critical coupling Kc = 2 * gamma = 2.0
    Kc = critical_point(c.pdf, loc=0.0)
    assert jnp.isclose(Kc, 2.0, atol=1e-4)


def test_orderparam_cauchy():
    c = Cauchy(loc=0.0, gamma=1.0)
    # For K <= Kc (Kc = 2.0), r = 0.0
    r_sub = orderparam(1.5, c)
    assert jnp.isclose(r_sub, 0.0, atol=1e-4)

    # For K > Kc, r = sqrt(1 - Kc / K)
    # K = 4.0, Kc = 2.0 -> r = sqrt(1 - 0.5) = sqrt(0.5) ~= 0.7071
    r_super = orderparam(4.0, c)
    assert jnp.isclose(r_super, jnp.sqrt(0.5), atol=1e-3)


def test_orderparam_normal_and_uniform():
    n = Normal(loc=0.0, scale=1.0)
    r_norm = orderparam(4.0, n)
    assert 0.0 <= r_norm <= 1.0

    u = Uniform(low=-1.0, high=1.0)
    r_uni = orderparam(4.0, u)
    assert 0.0 <= r_uni <= 1.0


def test_orderparam_with_distrax_distributions():
    import distrax

    # Native distrax.Normal
    dn = distrax.Normal(loc=0.0, scale=1.0)
    r_dn = orderparam(4.0, dn)
    assert 0.0 <= r_dn <= 1.0

    # Native distrax.Uniform
    du = distrax.Uniform(low=-1.0, high=1.0)
    r_du = orderparam(4.0, du)
    assert 0.0 <= r_du <= 1.0

    # Ensure results match jaxkuramoto classes
    n = Normal(loc=0.0, scale=1.0)
    assert jnp.isclose(r_dn, orderparam(4.0, n), atol=1e-5)



def test_ott_antonsen_cauchy():
    dist = Cauchy(loc=0.0, gamma=1.0)
    oa = OttAntonsen(dist=dist, K=4.0)
    init_z = jnp.array(0.1 + 0.0j)

    sol = odeint(
        oa.vector_fn,
        runge_kutta,
        0.0,
        5.0,
        0.05,
        init_z,
        observable_fn=oa.to_orderparam,
    )
    # Should converge towards theoretical order parameter sqrt(1 - 2/4) = 0.7071
    final_r = sol.observables[-1]
    assert jnp.isclose(final_r, jnp.sqrt(0.5), atol=0.05)


def test_ott_antonsen_cauchy_multiply():
    dist = CauchyMultiply(gamma1=1.0, gamma2=1.0, Omega=0.5)
    oa = OttAntonsen(dist=dist, K=3.0)
    init_zs = jnp.array([0.1 + 0.0j, 0.1 + 0.0j])

    sol = odeint(
        oa.vector_fn,
        runge_kutta,
        0.0,
        1.0,
        0.1,
        init_zs,
        observable_fn=oa.to_orderparam,
    )
    assert sol.observables.shape == (10,)
