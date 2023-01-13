from functools import partial
from jax.lax import while_loop
import jax.numpy as jnp
from jax import vjp, custom_vjp

@partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point(func, a, x_guess):
    def cond_fn(carry):
        x_prev, x = carry
        return jnp.abs(x - x_prev) > 1e-6
    def body_fn(carry):
        _, x = carry
        return x, func(a, x)
    _, x_star = while_loop(cond_fn, body_fn, (x_guess, func(a, x_guess)))
    return x_star

def fixed_point_fwd(func, a, x_init):
    x_star = fixed_point(func, a, x_init)
    return x_star, (a, x_star)

def rev_iter(f, packed, u):
    a, x_star, x_star_bar = packed
    _, vjp_x = vjp(lambda x: f(a, x), x_star)
    return x_star_bar + vjp_x(u)[0]

def fixed_point_bwd(func, res, x_star_bar):
    a, x_star = res
    _, vjp_a = vjp(lambda _a: func(_a, x_star), a)
    w = fixed_point(partial(rev_iter, func), (a, x_star, x_star_bar), x_star)
    a_bar, = vjp_a(w)
    return a_bar, jnp.zeros_like(x_star)

fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)