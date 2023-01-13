from functools import partial
import jax.numpy as jnp
from jaxkuramoto.solver import integral_fn, fixed_point

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
    def integrand_fn(x, a):
        return jnp.cos(x)**2 * pdf_fn(a * jnp.sin(x))
    return K * r * integral_fn(integrand_fn, K * r, -0.5*jnp.pi, 0.5*jnp.pi, n=n)

def self_consistent_rhs_uniform(r, K, gamma):
    phi = jnp.where(gamma>K*r, 0.5*jnp.pi, jnp.minimum(0.5*jnp.pi, jnp.arcsin(gamma/(K * r))))
    return 0.5 * r * K * (phi + 0.5 * jnp.sin(2 * phi)) / gamma

def orderparam(K, pdf_fn, n, r_guess=1.0, eps=1e-6):
    """Solve the self-consistent equation for the Kuramoto model and return the order parameter.

    Args:
        K: The coupling strength.
        pdf_fn: A function of the form pdf_fn(x) -> y.
        n: Number of trapezoids to use in the integral.
        r_guess: The initial guess for the order parameter.
        eps: The tolerance for the fixed point solver.

    Returns:
        Order parameter.
    """
    orderparam = fixed_point(partial(self_consistent_rhs, pdf_fn=pdf_fn, n=n), K, r_guess, eps)
    return orderparam

def orderparam_cauchy(K, gamma):
    """Return the order parameter for the Kuramoto model with Cauchy natural frequency distribution.

    Args:
        K: The coupling strength.
        gamma: Width of the Cauchy distribution. (p(x)=gamma/pi/(x^2+gamma^2)))

    Returns:
        Order parameter.
    """
    Kc = 2.0 * gamma
    return jnp.where(K > Kc, jnp.sqrt(1.0 - Kc / K), 0.0)

def orderparam_uniform(K, gamma, r_guess=1.0, eps=1e-6):
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
