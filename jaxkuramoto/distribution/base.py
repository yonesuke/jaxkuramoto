from typing import Sequence, Tuple, Union

import distrax
import jax.numpy as jnp
from jax import random
from jax.lax import while_loop


class Distribution(distrax.Distribution):
    """Base class for probability distributions in jaxkuramoto, inheriting from distrax.Distribution."""

    def __init__(self):
        super().__init__()
        self.symmetric = None
        self.unimodal = None
        self.interval = None
        self.y_max = None
        self._eps = 1e-6

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return ()

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Probability density function (alias to prob).

        Args:
            x (jnp.ndarray): Input values.

        Returns:
            jnp.ndarray: Probability density values.
        """
        return self.prob(x)

    def sample(
        self,
        key_or_seed: Union[random.PRNGKey, int] = None,
        shape_or_sample_shape: Union[int, Sequence[int]] = (),
        *,
        seed: Union[random.PRNGKey, int] = None,
        sample_shape: Union[int, Sequence[int]] = (),
    ) -> jnp.ndarray:
        """Sample from the distribution, supporting both jaxkuramoto and distrax calling conventions.

        Supports:
            - dist.sample(key, shape)
            - dist.sample(seed=key, sample_shape=shape)
        """
        # Determine the random key
        actual_seed = seed if seed is not None else key_or_seed
        if actual_seed is None:
            raise ValueError("A random key/seed must be provided to sample.")

        # Determine the sample shape
        if isinstance(shape_or_sample_shape, int):
            actual_shape = (shape_or_sample_shape,)
        elif len(shape_or_sample_shape) > 0:
            actual_shape = tuple(shape_or_sample_shape)
        elif isinstance(sample_shape, int):
            actual_shape = (sample_shape,)
        else:
            actual_shape = tuple(sample_shape)

        return super().sample(seed=actual_seed, sample_shape=actual_shape)

    def _sample_n(self, key: random.PRNGKey, n: int) -> jnp.ndarray:
        """Default fallback using sample method if overridden by subclasses."""
        raise NotImplementedError()

    def _rejection_sampling(self, key: random.PRNGKey, shape: Tuple[int, ...], x_min: float, x_max: float) -> jnp.ndarray:
        """Rejection sampling from a uniform proposal distribution.

        Args:
            key: A PRNGKey.
            shape: Shape of the samples to produce.
            x_min: Lower bound of the proposal distribution.
            x_max: Upper bound of the proposal distribution.

        Returns:
            Samples from the distribution.
        """
        n_sample = 1
        for s in shape:
            n_sample *= s

        def cond_fun(val):
            counter, _, _ = val
            return counter <= n_sample

        def body_fun(val):
            counter, curr_key, samples = val
            _u, _v = random.uniform(curr_key, (2,))
            u = x_min + (_u * (x_max - x_min))
            v = _v * self.y_max * 1.05
            flag = jnp.where(v <= self.pdf(u), 1, 0)
            samples = samples.at[counter * flag].set(u)
            counter += flag
            next_key = random.split(curr_key, num=2)[0]
            return counter, next_key, samples

        _, _, samples = while_loop(cond_fun, body_fun, (jnp.array(1), key, jnp.zeros((n_sample + 1,))))
        samples = samples[1:]
        return samples.reshape(shape)