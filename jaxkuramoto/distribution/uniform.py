import distrax
import jax.numpy as jnp
from jax import random

from .base import Distribution


class Uniform(Distribution):
    """Uniform distribution based on distrax.Uniform."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        super().__init__()
        if low >= high:
            raise ValueError("Low must be less than high.")
        self.low = low
        self.high = high
        self.symmetric = True
        self.unimodal = True
        self.interval = (low, high)
        self.loc = 0.5 * (low + high)
        self.scale = 0.5 * (high - low)
        self.y_max = 1.0 / (self.high - self.low)
        self._dist = distrax.Uniform(low=low, high=high)

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
