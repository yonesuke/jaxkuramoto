import jax.numpy as jnp
from jax import random

from .base import Distribution


class CauchyMultiply(Distribution):
    """Multiplied Cauchy distribution compliant with distrax.Distribution."""

    def __init__(self, Omega: float, gamma1: float, gamma2: float = 1.0):
        super().__init__()
        self.Omega = Omega
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.symmetric = (Omega == 0.0 and gamma1 == gamma2)
        self.unimodal = False
        self.interval = (-jnp.inf, jnp.inf)
        self.normalizer = gamma1 * gamma2 * ((gamma1 + gamma2) ** 2 + 4.0 * Omega ** 2) / jnp.pi / (gamma1 + gamma2)
        self.find_max()
        self._eps = jnp.minimum(1e-4, 0.5 / jnp.pi / (gamma1 + gamma2))
        sqrt_D = jnp.sqrt(self.normalizer / self._eps - 4.0 * Omega ** 2 * jnp.minimum(gamma1 ** 2, gamma2 ** 2))
        self.x_min = -jnp.sqrt(Omega ** 2 - gamma1 ** 2 + sqrt_D)
        self.x_max = -self.x_min

    def find_max(self):
        diff_poly_coeff = jnp.array([2.0, 0.0, self.gamma1 ** 2 + self.gamma2 ** 2 - 2 * self.Omega ** 2, self.Omega * (self.gamma1 ** 2 - self.gamma2 ** 2)])
        diff_poly_roots = jnp.roots(diff_poly_coeff)
        self.y_max = self.prob(diff_poly_roots.real).max()

    def _sample_n(self, key: random.PRNGKey, n: int) -> jnp.ndarray:
        return self._rejection_sampling(key, (n,), self.x_min, self.x_max)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return (
            jnp.log(self.normalizer)
            - jnp.log((value - self.Omega) ** 2 + self.gamma1 ** 2)
            - jnp.log((value + self.Omega) ** 2 + self.gamma2 ** 2)
        )

    def prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self.normalizer / ((value - self.Omega) ** 2 + self.gamma1 ** 2) / ((value + self.Omega) ** 2 + self.gamma2 ** 2)

