from typing import Callable

import jax.numpy as jnp

def critical_point(pdf_fn: Callable, loc: float = 0.0) -> float:
    r"""Find the critical coupling strength for the Kuramoto model.
    
    Args:
        pdf_fn (Callable): A function of the form :math:`x\mapsto p(x)`.
        loc (float): The center of the distribution.
    
    Returns:
        The critical coupling strength :math:`K_{\mathrm{c}}`.
    """
    return 2.0 / jnp.pi / pdf_fn(loc)