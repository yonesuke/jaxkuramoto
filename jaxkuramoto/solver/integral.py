from typing import Callable
from functools import partial
import jax.numpy as jnp
from jax import vmap, grad, custom_vjp

@partial(custom_vjp, nondiff_argnums=(0, 4))
def integral_fn(func, a, minval, maxval, n) -> float:
    """Integrate a function from minval to maxval using the trapezoidal rule.

    Args:
        func: A function of the form func(x, a) -> y.
        a: The parameter of the function.
        minval: The lower bound of the integral.
        maxval: The upper bound of the integral.
        n: The number of trapezoids to use.

    Returns:
        The value of the integral.
    """
    xs = jnp.linspace(minval, maxval, n+1)
    arr = vmap(lambda _x: func(_x, a))(xs)
    return jnp.trapz(arr, x=xs)

def integral_fwd(func, a, minval, maxval, n):
    integral_val = integral_fn(func, a, minval, maxval, n)
    return integral_val, (a, minval, maxval, n)

def integral_bwd(func, res, integral_bar):
    a, minval, maxval, n = res
    minval_bar = -integral_bar * func(minval, a)
    maxval_bar = integral_bar * func(maxval, a)
    f_grad = grad(func, argnums=1)
    a_bar = integral_bar * integral_fn(f_grad, a, minval, maxval, n)
    return a_bar, minval_bar, maxval_bar

integral_fn.defvjp(integral_fwd, integral_bwd)