import jax.numpy as jnp
from jax import random
from jax.scipy.special import betaln

from .base import Distribution

class FiniteDifferential(Distribution):
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
        self.interval = (-jnp.inf, jnp.inf)
        self.loc = loc
        self.scale = scale
        self.n = n
        self.x_min = self.loc - self.scale
        self.x_max = self.loc + self.scale
        self.interval = (self.x_min, self.x_max)
        self.normalizer = 1 / scale * jnp.exp(- betaln(n + 2, 0.5))
        self.y_max = self.normalizer

    def sample(self, key, shape):
        """Sample from the generalized Cauchy distribution.
        
        Args:
            key (jax.random.PRNGKey): Random number generator key.
            shape (tuple): Shape of the sample.
            
        Returns:
            jax.numpy.ndarray: Sampled values.
        """
        if self.n == 1:
            return self.loc + self.scale * random.cauchy(key, shape)
        else:
            return self._rejection_sampling(key, shape, self.x_min, self.x_max)
    
    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function.
        
        Args:
            x (jax.numpy.ndarray): Input values.

        Returns:
            jax.numpy.ndarray: Probability density values.
        """
        return jnp.where(
            (x >= self.x_min) * (x <= self.x_max), # check if x is in the interval
            self.normalizer * jnp.power(1 - ((x - self.loc) / self.scale) ** 2, self.n + 1), # true branch
            0 # false branch
        )