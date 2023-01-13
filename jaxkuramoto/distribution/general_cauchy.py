import jax.numpy as jnp
from jax import random

from .base import Distribution

class GeneralCauchy(Distribution):
    def __init__(self, loc: float = 0.0, gamma: float = 1.0, n: int = 1):
        """Generalized Cauchy distribution.

        Args:
            loc: Location parameter.
            gamma: Scale parameter.
            n: Degree of distribution (1 for Cauchy). p(x)=p(0)-C*x^(2n)+...

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
        self.loc = loc
        self.gamma = gamma
        self.n = n
        width = gamma * jnp.power(n*jnp.sin(0.5*jnp.pi/n)/jnp.pi/gamma/self._eps-1.0, 0.5 / self.n)
        self.x_min = self.loc - width
        self.x_max = self.loc + width
        self.y_max = n*jnp.sin(0.5*jnp.pi/n)/jnp.pi/gamma

    def sample(self, key, shape):
        """Sample from the generalized Cauchy distribution.
        
        Args:
            key (jax.random.PRNGKey): Random number generator key.
            shape (tuple): Shape of the sample.
            
        Returns:
            jax.numpy.ndarray: Sampled values.
        """
        if self.n == 1:
            return self.loc + self.gamma * random.cauchy(key, shape)
        else:
            return self._rejection_sampling(key, shape, self.x_min, self.x_max)

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function.
        
        Args:
            x (jax.numpy.ndarray): Input values.

        Returns:
            jax.numpy.ndarray: Probability density values.
        """
        return self.n * jnp.sin(0.5 * jnp.pi / self.n) / jnp.pi / self.gamma / (1.0 + (x / self.gamma) ** (2 * self.n))