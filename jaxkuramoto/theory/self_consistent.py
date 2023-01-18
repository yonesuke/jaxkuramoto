from functools import partial
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from jax.scipy.special import i0, i1
from jaxkuramoto.solver import integral_fn, fixed_point
from jaxkuramoto.distribution import Distribution

@partial(jit, static_argnums=(2, 3))
def self_consistent_rhs(r, K, pdf_fn, n):
    """Right-hand side of the self-consistent equation for the Kuramoto model.

    $$
    r = K * r * \int_0^{2\pi} cos^2(x) g(a * sin(x)) dx
    $$
    
    where $g$ is the probability density function of natural frequencies.

    Args:
        r: Order parameter.
        K: Coupling strength.
        pdf_fn: A function of the form pdf_fn(x) -> y.
        n: Number of trapezoids to use in the integral.

    Returns:
        Right-hand side of the self-consistent equation.
    """
    @jit
    def integrand_fn(x, a):
        return jnp.cos(x)**2 * pdf_fn(a * jnp.sin(x))
    return K * r * integral_fn(integrand_fn, K * r, -0.5*jnp.pi, 0.5*jnp.pi, n=n)

@jit
def self_consistent_rhs_uniform(r, K, gamma):
    phi = cond(
        gamma > K * r, # condition
        lambda _: 0.5 * jnp.pi, # true branch
        lambda _: jnp.minimum(0.5 * jnp.pi, jnp.arcsin(gamma / (K * r))), # false branch
        None # operand
    )
    return 0.5 * r * K * (phi + 0.5 * jnp.sin(2 * phi)) / gamma

@jit
def self_consistent_rhs_normal(r, K, sigma):
    i0_val, i1_val = i0(0.25 * K * K * r * r / sigma / sigma), i1(0.25 * K * K * r * r / sigma / sigma)
    return 0.5 * r * K / sigma * jnp.sqrt(0.5 * jnp.pi) * jnp.exp(-0.25 * r**2 * K**2 / sigma**2) * (i0_val + i1_val)

def orderparam(K, dist: Distribution, n=10**3, r_guess=1.0, eps=1e-6):
    """Solve the self-consistent equation for the Kuramoto model and return the order parameter.

    Args:
        K: The coupling strength.
        dist: A function of the form pdf_fn(x) -> y.
        n: Number of trapezoids to use in the integral.
        r_guess: The initial guess for the order parameter.
        eps: The tolerance for the fixed point solver.

    Returns:
        Order parameter.
    """
    if (not dist.symmetric) or (not dist.unimodal):
        raise ValueError("Distribution must be symmetric and unimodal.")
    if dist.__class__.__name__ == "Uniform":
        return _orderparam_uniform(K, dist.scale, r_guess, eps)
    elif dist.__class__.__name__ == "Cauchy":
        return _orderparam_cauchy(K, dist.gamma)
    elif dist.__class__.__name__ == "Normal":
        return _orderparam_normal(K, dist.scale, r_guess, eps)
    else:
        pdf_centered = lambda x: dist.pdf(x - dist.loc)
        return fixed_point(partial(self_consistent_rhs, pdf_fn=pdf_centered, n=n), K, r_guess, eps)

@jit
def _orderparam_cauchy(K, gamma):
    """Return the order parameter for the Kuramoto model with Cauchy natural frequency distribution.

    Args:
        K: The coupling strength.
        gamma: Width of the Cauchy distribution. (p(x)=gamma/pi/(x^2+gamma^2)))

    Returns:
        Order parameter.
    """
    Kc = 2.0 * gamma
    return cond(
        K > Kc, # condition
        lambda _: jnp.sqrt(1.0 - Kc / K), # true branch
        lambda _: 0.0, # false branch
        None # operand
    )

@jit
def _orderparam_normal(K, sigma, r_guess=1.0, eps=1e-6):
    """Solve the self-consistent equation for the Kuramoto model with normal natural frequency distribution and return the order parameter.

    Args:
        K: The coupling strength.
        sigma: Standard deviation of the natural frequency.
        r_guess: The initial guess for the order parameter.
        eps: The tolerance for the fixed point solver.

    Returns:
        Order parameter.
    """
    orderparam = fixed_point(partial(self_consistent_rhs_normal, sigma=sigma), K, r_guess, eps)
    return orderparam

@jit
def _orderparam_uniform(K, gamma, r_guess=1.0, eps=1e-6):
    """Solve the self-consistent equation for the Kuramoto model with uniform natural frequency distribution and return the order parameter.

    Args:
        K: The coupling strength.
        gamma: Width of the natural frequency. (0.5 / gamma for |x|<gamma)
        r_guess: The initial guess for the order parameter.
        eps: The tolerance for the fixed point solver.

    Returns:
        Order parameter.
    """
    orderparam = fixed_point(partial(self_consistent_rhs_uniform, gamma=gamma), K, r_guess, eps)
    return orderparam
