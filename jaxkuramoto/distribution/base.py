import jax.numpy as jnp
from jax import random
from jax.lax import while_loop

class Distribution:
    def __init__(self):
        self.symmetric = None
        self.unimodal = None
        self.interval = None
        self.y_max = None
        self._eps = 1e-6

    def sample(self, key, shape):
        raise NotImplementedError()

    def log_prob(self, x):
        raise NotImplementedError()

    def pdf(self, x):
        raise NotImplementedError()

    def _rejection_sampling(self, key, shape, x_min, x_max):
        """Rejection sampling from a uniform distribution.
        
        Args:
            key: A PRNGKey.
            n: Number of samples.
            x_min: Lower bound of the uniform distribution.
            x_max: Upper bound of the uniform distribution.

        Returns:
            Samples from the uniform distribution.
        """
        n_sample = 1
        for s in shape:
            n_sample *= s
        def cond_fun(val):
            counter, _, _ = val
            return counter <= n_sample
        def body_fun(val):
            counter, key, samples = val
            _u, _v = random.uniform(key, (2,))
            u = x_min + (_u * (x_max - x_min))
            v = _v * self.y_max * 1.05
            flag = jnp.where(v <= self.pdf(u), 1, 0)
            samples = samples.at[counter * flag].set(u)
            counter += flag
            next_key = random.split(key, num=1)[0]
            return counter, next_key, samples
        _, _, samples = while_loop(cond_fun, body_fun, (jnp.array(1), key, jnp.zeros((n_sample+1,))))
        samples = samples[1:]
        return samples.reshape(shape)