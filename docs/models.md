# List of predefined models

`jaxkuramoto` provides implementations of widely studied Kuramoto-type oscillator models.

## Kuramoto model

The standard Kuramoto model is defined by:

$$
\frac{\mathrm{d}\theta_{i}}{\mathrm{d} t} = \omega_{i} + \frac{K}{N}\sum_{j=1}^{N}\sin(\theta_{j}-\theta_{i})
$$

### Mean-field reformulation
Directly summing over all pairs requires $\mathcal{O}(N^2)$ operations. In `jaxkuramoto`, we rewrite the interaction using the complex order parameter:

$$
r e^{\mathrm{i}\psi} = \frac{1}{N}\sum_{j=1}^{N} e^{\mathrm{i}\theta_j} = r_x + \mathrm{i} r_y
$$

where $r_x = \frac{1}{N}\sum_{j=1}^N \cos\theta_j$ and $r_y = \frac{1}{N}\sum_{j=1}^N \sin\theta_j$.
The equation of motion then simplifies to $\mathcal{O}(N)$ computation:

$$
\frac{\mathrm{d}\theta_{i}}{\mathrm{d} t} = \omega_{i} + K (r_y \cos\theta_i - r_x \sin\theta_i)
$$

### Example usage
```python
import jax.numpy as jnp
from jaxkuramoto import Kuramoto, odeint
from jaxkuramoto.solver import runge_kutta

omegas = jnp.linspace(-1.0, 1.0, 100)
model = Kuramoto(omegas=omegas, K=2.0)

init_thetas = jnp.zeros(100)
sol = odeint(model.vector_fn, runge_kutta, 0.0, 10.0, 0.05, init_thetas, observable_fn=model.orderparameter)
```

---

## Sakaguchi-Kuramoto model

The **Sakaguchi-Kuramoto model** introduces a constant phase lag $\alpha \in (-\pi/2, \pi/2)$ to the interaction term:

$$
\frac{\mathrm{d}\theta_{i}}{\mathrm{d} t} = \omega_{i} + \frac{K}{N}\sum_{j=1}^{N}\sin(\theta_{j}-\theta_{i} - \alpha)
$$

The phase lag $\alpha$ breaks the gradient system property and can induce complex collective dynamics such as frequency pulling, collective oscillation, and frustration.

### Example usage
```python
import jax.numpy as jnp
from jaxkuramoto import SakaguchiKuramoto, odeint
from jaxkuramoto.solver import runge_kutta

omegas = jnp.linspace(-1.0, 1.0, 100)
alpha = 0.3  # phase shift parameter
model = SakaguchiKuramoto(omegas=omegas, K=2.0, alpha=alpha)

init_thetas = jnp.zeros(100)
sol = odeint(model.vector_fn, runge_kutta, 0.0, 10.0, 0.05, init_thetas)
```