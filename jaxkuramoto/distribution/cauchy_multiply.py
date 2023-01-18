import jax.numpy as jnp
from jax import random, grad

from .base import Distribution

class CauchyMultiply(Distribution):
    def __init__(self, Omega: float, gamma1: float, gamma2: float = 1.0):
        super().__init__()
        self.Omega = Omega
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.interval = (-jnp.inf, jnp.inf)
        self.normalizer = gamma1*gamma2*((gamma1+gamma2)**2+4.0*Omega**2)/jnp.pi/(gamma1+gamma2)
        self.find_max()
        self._eps = jnp.minimum(1e-4, 0.5 / jnp.pi / (gamma1 + gamma2))
        sqrt_D = jnp.sqrt(self.normalizer / self._eps - 4.0 * Omega ** 2 * jnp.minimum(gamma1**2, gamma2**2))
        self.x_min = -jnp.sqrt(Omega**2-gamma1**2+sqrt_D)
        self.x_max = -self.x_min


    def find_max(self):
        diff_poly_coeff = jnp.array([2.0, 0.0, self.gamma1**2+self.gamma2**2-2*self.Omega**2, self.Omega*(self.gamma1**2-self.gamma2**2)])
        diff_poly_roots = jnp.roots(diff_poly_coeff)
        self.y_max = self.pdf(diff_poly_roots.real).max()

    def sample(self, key, shape):
        """Sample from the Mutiplied Cauchy distribution.
        
        Args:
            key (jax.random.PRNGKey): Random number generator key.
            shape (tuple): Shape of the sample.
            
        Returns:
            jax.numpy.ndarray: Sampled values.
        """
        return self._rejection_sampling(key, shape, self.x_min, self.x_max)

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function.
        
        Args:
            x (jax.numpy.ndarray): Input values.
            
        Returns:
            jax.numpy.ndarray: Probability density values.
        """
        return self.normalizer/((x-self.Omega)**2+self.gamma1**2)/((x+self.Omega)**2+self.gamma2**2)
