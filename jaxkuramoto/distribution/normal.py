import jax.numpy as jnp
from jax import random

from .base import Distribution

class Normal(Distribution):
    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        """Normal distribution.

        Args:
            loc: Location parameter. center = loc.
            scale: Scale parameter. variance = scale ** 2.

        Raises:
            ValueError: If `scale` is not positive.

        References:
            https://en.wikipedia.org/wiki/Normal_distribution
        """
        super().__init__()
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        self.symmetric = True
        self.unimodal = True
        self.loc = loc
        self.scale = scale
        self.y_max = 1.0 / jnp.sqrt(2 * jnp.pi) / self.scale

    def sample(self, key, shape):
        """Sample from the Normal distribution.
        
        Args:
            key (jax.random.PRNGKey): Random number generator key.
            shape (tuple): Shape of the sample.
            
        Returns:
            jax.numpy.ndarray: Sampled values.
        """
        return self.loc + self.scale * random.normal(key, shape)

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function.
        
        Args:
            x (jax.numpy.ndarray): Input values.

        Returns:
            jax.numpy.ndarray: Probability density values.
        """
        return jnp.exp(-0.5 * ((x - self.loc) / self.scale) ** 2) / jnp.sqrt(2 * jnp.pi) / self.scale