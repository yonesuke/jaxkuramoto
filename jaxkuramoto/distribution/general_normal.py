import jax.numpy as jnp
from jax import random
import math

from .base import Distribution

class GeneralNormal(Distribution):
    def __init__(self, loc: float = 0.0, gamma: float = 1.0, n: int = 1):
        """Generalized normal distribution.

        Args:
            loc: Location parameter.
            gamma: Scale parameter.
            n: Degree of distribution (1 for normal). p(x)=p(0)-C*x^(2n)+...

        Raises:
            ValueError: If `gamma` is not positive.
            ValueError: If `n` is not positive.
        """
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

    def sample(self, key, shape):
        """Sample from the generalized normal distribution.
        
        Args:
            key (jax.random.PRNGKey): Random number generator key.
            shape (tuple): Shape of the sample.
            
        Returns:
            jax.numpy.ndarray: Sampled values.
        """
        if self.n == 1:
            scale = 1.0 / self.gamma / jnp.sqrt(2.0 * jnp.pi)
            return self.loc + scale * random.normal(key, shape)
        else:
            return self._rejection_sampling(key, shape, self.x_min, self.x_max)

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function.
        
        Args:
            x (jax.numpy.ndarray): Input values.

        Returns:
            jax.numpy.ndarray: Probability density values.
        """
        return self.n * self.gamma * jnp.exp(-(self.gamma * x)**(2*self.n)) / math.gamma(0.5 / self.n)