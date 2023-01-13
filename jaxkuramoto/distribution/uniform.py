import jax.numpy as jnp
from jax import random

from .base import Distribution

class Uniform(Distribution):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        """Uniform distribution.
        
        Args:
            low: Lower bound.
            high: Upper bound.

        Raises:
            ValueError: If `low` is not less than `high`.
        """
        super().__init__()
        if low >= high:
            raise ValueError("Low must be less than high.")
        self.symmetric = True
        self.unimodal = True
        self.low = low
        self.high = high
        self.loc = 0.5 * (low + high)
        self.scale = 0.5 * (high - low)
        self.y_max = 1.0 / (self.high - self.low)

    def sample(self, key, shape):
        """Sample from the Uniform distribution.
        
        Args:
            key (jax.random.PRNGKey): Random number generator key.
            shape (tuple): Shape of the sample.
            
        Returns:
            jax.numpy.ndarray: Sampled values.
        """
        return self.low + (self.high - self.low) * random.uniform(key, shape)

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function.
        
        Args:
            x (jax.numpy.ndarray): Input values.

        Returns:
            jax.numpy.ndarray: Probability density values.
        """
        return jnp.where((x >= self.low) & (x <= self.high), self.y_max, 0.0)
