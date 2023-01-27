# How to use `distribution`

In `jaxkuramoto`, we provide a class `Distribution` to deal with distributions. The class has the `sample` method, so you can sample from the distribution as follows.

```python
import jax; jax.config.update("jax_enable_x64", True)
from jax import random
from jaxkuramoto.distribution import Cauchy

n_sample = 100
dist = Cauchy(0.0, 1.0)

seed = 0; key = random.PRNGKey(seed)
samples = dist.sample(key, (n_sample,))
```

The `sample` method takes a `PRNGKey` and a shape of samples as arguments. The shape of samples is a tuple of integers. The `sample` method returns a `DeviceArray` of samples.

We prepare some distributions in `jaxkuramoto.distribution`. The distributions are as follows.
- `Normal`: The normal distribution.
- `Cauchy`: The Cauchy distribution.
- `Uniform`: The uniform distribution.
- `GeneralNormal`: The generalized normal distribution.
- `GeneralCauchy`: The generalized Cauchy distribution.
- `CauchyMultiply`: The product of two Cauchy distributions.
- `FiniteDifferential`: The finite-differentiable distribution.

Check out [List of distributions](distributions) for more details.


## Define your own distribution

You can define your own distribution by inheriting the `Distribution` class.
We explain the attributes of the `Distribution` class below.

- `symmetric`: If the distribution is symmetric, set `True`. If not, set `False`.
- `unimodal`: If the distribution is unimodal, set `True`. If not, set `False`.
- `interval`: The interval of the distribution. The interval is a tuple of two floats, `(a, b)`, where `a` and `b` are the minimum and maximum values of the distribution, respectively. If the distribution has infinite support, set `(-jnp.inf, jnp.inf)`.
- `y_max`: The maximum value of the distribution.
- `_eps`: The tolerance for numerical calculation for the rejection sampling. If you want to use the rejection sampling method, you have to set this value in advance. For a distribution with infinite support, you have to set this value sufficiently small.
- `x_min`: The minimum value of the distribution. If you want to use the rejection sampling method, you have to set this value in advance. For a distribution with infinite support, you have to set this value manually sufficiently large that the probability of sampling a value is sufficiently small.
- `x_max`: The maximum value of the distribution. If you want to use the rejection sampling method, you have to set this value in advance. For a distribution with infinite support, you have to set this value manually sufficiently large that the probability of sampling a value is sufficiently small.

To implement the sampling method, you have the following two options.

- First option: Use the [rejection sampling method](https://en.wikipedia.org/wiki/Rejection_sampling). We internally make a rejcetion sampling method function for you. Write just like this:
    ```python
    def sample(self, key, shape):
        return self._rejection_sampling(key, shape, self.x_min, self.x_max)
    ```
    `self.x_min` and `self.x_max` are the minimum and maximum values of the distribution, respectively. If you want to use the rejection sampling method, you have to set these values in advance. For a distribution with infinite support, you have to set these values sufficiently large.
- Second option: Use the [inverse transform sampling method](https://en.wikipedia.org/wiki/Inverse_transform_sampling). If you know the inverse of the cumulative distribution function (CDF) of the distribution, you can use this method.
    ```python
    from jax import random
    def inv_cdf(self, x):
        # inverse of the cumulative distribution function
        # write your code here
        return inv_cdf
    def sample(self, key, shape):
        rv_uniform = random.uniform(key, shape)
        return self.inv_cdf(rv_uniform)
    ```

We show an example of a custom distribution class below.
```python
import jax; jax.config.update("jax_enable_x64", True)
from jaxkuramoto.distribution import Distribution

class MyDistribution(Distribution):
    """My distribution"""
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.symmetric = True # if the distribution is symmetric
        self.unimodal = True # if the distribution is unimodal
        self.interval = (-jnp.inf, jnp.inf) # the interval of the distribution
        self.y_max = None # the maximum value of the distribution
        self._eps = 1e-6 # the tolerance for numerical calculation for the rejection sampling
        self.x_min = -1.0 # the minimum value of the distribution
        self.x_max = 1.0 # the maximum value of the distribution

    def sample(self, key, shape):
        return self._rejection_sampling(key, shape, self.x_min, self.x_max)

    def pdf(self, x):
        # probability desnity function
        # write your code here
        return pdf
```