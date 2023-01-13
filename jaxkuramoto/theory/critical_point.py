import jax.numpy as jnp

def critical_point(pdf_fn, loc=0.0):
    """Find the critical coupling strength for the Kuramoto model.
    
    Args:
        `pdf_fn`: A function of the form pdf_fn(x) -> y.
        `loc`: The center of the distribution.
    
    Returns:
        Critical coupling strength Kc.

    Notes:
        `pdf_fn` must be normalized, symmetric, and have a single maximum at `loc`.
    """
    return 2.0 / jnp.pi / pdf_fn(loc)