import jax.numpy as jnp
from jax import random
from jax.scipy.special import betaln

from .base import Distribution


class FiniteDifferential(Distribution):
    """Finite differential distribution compliant with distrax.Distribution."""

    def __init__(self, loc: float = 0.0, scale: float = 1.0, n: int = 1):
        super().__init__()
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        if not isinstance(n, int):
            raise ValueError("N must be integer.")
        if n <= 0:
            raise ValueError("N must be positive.")
        self.symmetric = True
        self.unimodal = True
        self.loc = loc
        self.scale = scale
        self.n = n
        self.x_min = self.loc - self.scale
        self.x_max = self.loc + self.scale
        self.interval = (self.x_min, self.x_max)
        self.normalizer = 1.0 / scale * jnp.exp(-betaln(n + 2, 0.5))
        self.y_max = self.normalizer

    def _sample_n(self, key: random.PRNGKey, n: int) -> jnp.ndarray:
        return self._rejection_sampling(key, (n,), self.x_min, self.x_max)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        standardized = (value - self.loc) / self.scale
        in_support = (value >= self.x_min) & (value <= self.x_max)
        safe_std = jnp.where(in_support, standardized, 0.0)
        log_p = -jnp.log(self.scale) - betaln(self.n + 2, 0.5) + (self.n + 1) * jnp.log1p(-safe_std ** 2)
        return jnp.where(in_support, log_p, -jnp.inf)

    def prob(self, value: jnp.ndarray) -> jnp.ndarray:
        standardized = (value - self.loc) / self.scale
        in_support = (value >= self.x_min) & (value <= self.x_max)
        safe_std = jnp.where(in_support, standardized, 0.0)
        p = self.normalizer * ((1.0 - safe_std ** 2) ** (self.n + 1))
        return jnp.where(in_support, p, 0.0)