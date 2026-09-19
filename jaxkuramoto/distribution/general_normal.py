import math

import jax.numpy as jnp
from jax import random

from .base import Distribution


class GeneralNormal(Distribution):
    """Generalized normal distribution compliant with distrax.Distribution."""

    def __init__(self, loc: float = 0.0, gamma: float = 1.0, n: int = 1):
        super().__init__()
        if gamma <= 0:
            raise ValueError("Gamma must be positive.")
        if not isinstance(n, int):
            raise ValueError("N must be integer.")
        if n <= 0:
            raise ValueError("N must be positive.")
        self.symmetric = True
        self.unimodal = True
        self.interval = (-jnp.inf, jnp.inf)
        self.loc = loc
        self.gamma = gamma
        self.n = n
        self.y_max = self.n * self.gamma / math.gamma(0.5 / self.n)
        width = jnp.power(jnp.log(n * gamma / self._eps / math.gamma(0.5 / self.n)), 0.5 / self.n) / self.gamma
        self.x_min = self.loc - width
        self.x_max = self.loc + width

    def _sample_n(self, key: random.PRNGKey, n: int) -> jnp.ndarray:
        if self.n == 1:
            scale = 1.0 / self.gamma / jnp.sqrt(2.0 * jnp.pi)
            return self.loc + scale * random.normal(key, (n,))
        else:
            return self._rejection_sampling(key, (n,), self.x_min, self.x_max)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        log_norm = jnp.log(self.n * self.gamma) - math.lgamma(0.5 / self.n)
        return log_norm - (self.gamma * (value - self.loc)) ** (2 * self.n)

    def prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self.n * self.gamma * jnp.exp(- (self.gamma * (value - self.loc)) ** (2 * self.n)) / math.gamma(0.5 / self.n)