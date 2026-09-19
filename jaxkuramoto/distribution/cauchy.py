import jax.numpy as jnp
from jax import random

from .base import Distribution


class Cauchy(Distribution):
    """Cauchy (Lorentzian) distribution compliant with distrax.Distribution."""

    def __init__(self, loc: float = 0.0, gamma: float = None, scale: float = None):
        super().__init__()
        actual_scale = scale if scale is not None else (gamma if gamma is not None else 1.0)
        if actual_scale <= 0:
            raise ValueError("Scale/gamma must be positive.")
        self.loc = loc
        self.gamma = actual_scale
        self.scale = actual_scale
        self.symmetric = True
        self.unimodal = True
        self.interval = (-jnp.inf, jnp.inf)
        self.y_max = 1.0 / jnp.pi / self.gamma

    def _sample_n(self, key: random.PRNGKey, n: int) -> jnp.ndarray:
        return self.loc + self.gamma * random.cauchy(key, (n,))

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        standardized = (value - self.loc) / self.gamma
        return -jnp.log(jnp.pi * self.gamma) - jnp.log1p(standardized ** 2)

    def prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self.gamma / (jnp.pi * (self.gamma ** 2 + (value - self.loc) ** 2))