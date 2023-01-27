# How to use `odeint`

The function `odeint` is to solve ordinary differential equations (ODEs) with the following form:

$$
\frac{\mathrm{d}x}{\mathrm{d}t} = f(t, x)
$$

```python
from jaxkuramoto import odeint

sol = odeint(vector_fn, solver, t0, t1, dt, init_state, observable_fn)
```

## Arguments

- `vector_fn`: A function that takes the current time and state and returns the derivative of the state. For ODE with $\dot{x}=f(t,x)$, `vector_fn` is $f$.
    ```python
    def vector_fn(t, state):
        return derivative
    ```

- `solver`: A solver algorithm to solve the ODE. The following solvers are available. 

    - `euler`: [Euler's method](https://en.wikipedia.org/wiki/Euler_method).
    - `runge_kutta`: [Runge-Kutta method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods).

    Check out [list of solvers](ode_solver) for more details. We are planning to add more solvers in the future, especially adaptive time step solvers.

- `t0`: The initial time.

- `t1`: The final time.

- `dt`: The time step.

- `init_state`: The initial state.

- `observable_fn`: A function that takes the current time and state and returns the observable.
Default is `None`. If `None`, `observable_fn` returns the current state.

    ```python
    def observable_fn(t, state):
        return state
    ```

    See [here](#why-observable-fn) for more details.

## Returns

The function `odeint` returns a `Solution` object that contains the solution of the ODE.
It has the following attributes:

- `t0`: The initial time.

- `t1`: The final time.

- `ts`: The time points.

- `initial_state`: The initial state.

- `final_state`: The final state.

- `observables`: The observables at the time points.

## Example

We provide some examples of using `odeint` in the following.
Though `jaxkuramoto` is designed for solving Kuramoto models, it can be used to solve any ODEs if you want to.

### van der Pol oscillator

The first example is the [van der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator).

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxkuramoto import odeint
from jaxkuramoto.solver import runge_kutta

def vdp_fn(t, state, mu):
    x, y = state
    return jnp.array([y, -x + mu * (1 - x**2) * y])

mu = 1.0
t0, t1, dt = 0, 100, 0.01
init_state = jnp.array([2.0, 0.0])

sol = odeint(
    lambda t, state: vdp_fn(t, state, mu),
    runge_kutta, t0, t1, dt, init_state
)
```


### Kuramoto model

Regarding the Kuramoto model, we provide a class `Kuramoto` to deal with it.
The class has the `vector_fn` and `orderparameter`, so you can use `odeint` as follows.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jaxkuramoto import odeint, Kuramoto
from jaxkuramoto.distribution import Cauchy
from jaxkuramoto.solver import runge_kutta

n_oscillator = 100
K = 1.0
omegas = Cauchy(0.0, 1.0).sample(random.PRNGKey(0), (n_oscillator,))

model = Kuramoto(omegas, K)
init_state = random.uniform(random.PRNGKey(1), (n_oscillator,)) * 2 * jnp.pi
sol = odeint(
    model.vector_fn,
    runge_kutta, 0, 100, 0.01, init_state,
    observable_fn=model.orderparameter
)
```

Check out [Kuramoto model](examples/kuramoto) for more details, and explore the miracle of synchronization!!

## Notes

### Deal with parameters

If your ODE depends on parameters, you can use `lambda` to deal with them.
The van der Pol oscillator illustrated in the previous section is a good example.

```python
from functools import partial
from jaxkuramoto import odeint

def vector_fn(t, state, arg1, arg2):
    ...

def observable_fn(t, state, arg1, arg2):
    ...

arg1, arg2 = ...

sol = odeint(
    # use lambda to make the vector_fn in a form that odeint can accept
    lambda t, state: vector_fn(t, state, arg1, arg2), 
    solver, t0, t1, dt, init_state,
    lambda t, state: observable_fn(t, state, arg1, arg2) # same as above
)
```

### Why `observable_fn`?

When solving the Kuramoto model, the number of oscillators is usually large.
Thus, it is not efficient to store the state of all oscillators at each time point.
Instead, we can store the observable of the state at each time point.
For example, in the Kuramoto model, the observable is often set to the order parameter, defined as

$$
r = \left|\frac{1}{N} \sum_{i=1}^N \exp(i \theta_i)\right|
$$

Otherwise, if you want to store the state of all oscillators at each time point, you do not have to specify `observable_fn`. The default is `None` and it stores the state of all oscillators at each time point.