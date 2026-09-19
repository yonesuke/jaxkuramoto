import jax
import jax.numpy as jnp
import pytest

from jaxkuramoto.distribution import (
    Cauchy,
    CauchyMultiply,
    FiniteDifferential,
    GeneralCauchy,
    GeneralNormal,
    Normal,
    Uniform,
)


def test_normal_distribution():
    dist = Normal(loc=0.0, scale=1.0)
    key = jax.random.PRNGKey(0)
    samples = dist.sample(key, (100,))
    assert samples.shape == (100,)
    pdf_val = dist.pdf(0.0)
    assert jnp.isclose(pdf_val, 1.0 / jnp.sqrt(2.0 * jnp.pi), atol=1e-4)


def test_cauchy_distribution():
    dist = Cauchy(loc=0.0, gamma=1.0)
    key = jax.random.PRNGKey(1)
    samples = dist.sample(key, (50,))
    assert samples.shape == (50,)
    pdf_val = dist.pdf(0.0)
    assert jnp.isclose(pdf_val, 1.0 / jnp.pi, atol=1e-4)


def test_uniform_distribution():
    dist = Uniform(low=-2.0, high=2.0)
    key = jax.random.PRNGKey(2)
    samples = dist.sample(key, (50,))
    assert samples.shape == (50,)
    assert jnp.all((samples >= -2.0) & (samples <= 2.0))
    pdf_val = dist.pdf(0.0)
    assert jnp.isclose(pdf_val, 0.25, atol=1e-4)


def test_general_normal():
    dist = GeneralNormal(loc=0.0, gamma=1.0, n=2)
    key = jax.random.PRNGKey(3)
    samples = dist.sample(key, (20,))
    assert samples.shape == (20,)
    pdf_val = dist.pdf(0.0)
    assert pdf_val > 0.0


def test_general_cauchy():
    dist = GeneralCauchy(loc=0.0, gamma=1.0, n=2)
    key = jax.random.PRNGKey(4)
    samples = dist.sample(key, (20,))
    assert samples.shape == (20,)
    pdf_val = dist.pdf(0.0)
    assert pdf_val > 0.0


def test_cauchy_multiply():
    dist = CauchyMultiply(gamma1=1.0, gamma2=1.0, Omega=0.5)
    key = jax.random.PRNGKey(5)
    samples = dist.sample(key, (20,))
    assert samples.shape == (20,)
    pdf_val = dist.pdf(0.0)
    assert pdf_val > 0.0


def test_finite_differential():
    dist = FiniteDifferential(loc=0.0, scale=1.0, n=1)
    pdf_val = dist.pdf(0.0)
    assert pdf_val > 0.0
