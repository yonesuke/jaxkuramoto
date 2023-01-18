import jax.numpy as jnp
from jax import random

from .base import Distribution

class Cauchy(Distribution):
    def __init__(self, loc: float = 0.0, gamma: float = 1.0):
        """Cauchy distribution.

        Args:
            loc: Location parameter.
            gamma: Scale parameter.

        Raises:
            ValueError: If `gamma` is not positive.

        References:
            https://en.wikipedia.org/wiki/Cauchy_distribution
        """
        super().__init__()
        if gamma <= 0:
            raise ValueError("Gamma must be positive.")
        self.symmetric = True
        self.unimodal = True
        self.interval = (-jnp.inf, jnp.inf)
        self.loc = loc
        self.gamma = gamma
        self.y_max = 1.0 / jnp.pi / self.gamma

    def sample(self, key, shape):
        """Sample from the Cauchy distribution.
        
        Args:
            key (jax.random.PRNGKey): Random number generator key.
            shape (tuple): Shape of the sample.
            
        Returns:
            jax.numpy.ndarray: Sampled values.
        """
        return self.loc + self.gamma * random.cauchy(key, shape)

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function.
        
        Args:
            x (jax.numpy.ndarray): Input values.

        Returns:
            jax.numpy.ndarray: Probability density values.
        """
        return self.gamma / jnp.pi / (self.gamma ** 2 + (x - self.loc) ** 2)