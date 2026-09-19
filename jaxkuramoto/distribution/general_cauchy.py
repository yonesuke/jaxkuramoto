import jax.numpy as jnp
from jax import random

from .base import Distribution


class GeneralCauchy(Distribution):
    """Generalized Cauchy distribution compliant with distrax.Distribution."""

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
        self.y_max = n * jnp.sin(0.5 * jnp.pi / n) / jnp.pi / gamma
        width = gamma * jnp.power(n * jnp.sin(0.5 * jnp.pi / n) / jnp.pi / gamma / self._eps - 1.0, 0.5 / self.n)
        self.x_min = self.loc - width
        self.x_max = self.loc + width

    def _sample_n(self, key: random.PRNGKey, n: int) -> jnp.ndarray:
        if self.n == 1:
            return self.loc + self.gamma * random.cauchy(key, (n,))
        else:
            return self._rejection_sampling(key, (n,), self.x_min, self.x_max)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        norm = self.n * jnp.sin(0.5 * jnp.pi / self.n) / jnp.pi / self.gamma
        standardized = (value - self.loc) / self.gamma
        return jnp.log(norm) - jnp.log1p(standardized ** (2 * self.n))

    def prob(self, value: jnp.ndarray) -> jnp.ndarray:
        standardized = (value - self.loc) / self.gamma
        return self.n * jnp.sin(0.5 * jnp.pi / self.n) / jnp.pi / self.gamma / (1.0 + standardized ** (2 * self.n))