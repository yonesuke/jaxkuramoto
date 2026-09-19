import distrax
import jax.numpy as jnp
from jax import random

from .base import Distribution


class Normal(Distribution):
    """Normal (Gaussian) distribution based on distrax.Normal."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        super().__init__()
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        self.loc = loc
        self.scale = scale
        self.symmetric = True
        self.unimodal = True
        self.interval = (-jnp.inf, jnp.inf)
        self.y_max = 1.0 / jnp.sqrt(2 * jnp.pi) / self.scale
        self._dist = distrax.Normal(loc=loc, scale=scale)

    @property
    def event_shape(self):
        return self._dist.event_shape

    @property
    def batch_shape(self):
        return self._dist.batch_shape

    def _sample_n(self, key: random.PRNGKey, n: int) -> jnp.ndarray:
        return self._dist._sample_n(key, n)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self._dist.log_prob(value)

    def prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self._dist.prob(value)