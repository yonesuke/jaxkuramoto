# Package Reference

This page provides the API reference for `jaxkuramoto`.

---

## Models

### `jaxkuramoto.Kuramoto`
```python
class Kuramoto(ODE):
    def __init__(self, omegas: jnp.ndarray, K: float) -> None
```
Standard all-to-all coupled Kuramoto model.

* **Parameters**:
  * `omegas` (*jnp.ndarray*): Intrinsic natural frequencies of oscillators $\omega_i$.
  * `K` (*float*): Global coupling strength.

* **Methods**:
  * `vector_fn(t: float, thetas: jnp.ndarray) -> jnp.ndarray`: Returns the vector field $\dot{\theta}_i = \omega_i + K (r_y \cos\theta_i - r_x \sin\theta_i)$.
  * `orderparameter(t: float, thetas: jnp.ndarray) -> float`: Calculates the complex order parameter magnitude $r = |\frac{1}{N}\sum_j e^{\mathrm{i}\theta_j}|$.

---

### `jaxkuramoto.SakaguchiKuramoto`
```python
class SakaguchiKuramoto(ODE):
    def __init__(self, omegas: jnp.ndarray, K: float, alpha: float) -> None
```
Sakaguchi-Kuramoto model with constant phase lag $\alpha$.

* **Parameters**:
  * `omegas` (*jnp.ndarray*): Intrinsic natural frequencies of oscillators $\omega_i$.
  * `K` (*float*): Coupling strength.
  * `alpha` (*float*): Phase shift parameter $\alpha \in (-\pi/2, \pi/2)$.

* **Methods**:
  * `vector_fn(t: float, thetas: jnp.ndarray) -> jnp.ndarray`: Returns the vector field with phase lag.
  * `orderparameter(t: float, thetas: jnp.ndarray) -> float`: Returns the order parameter magnitude.

---

## ODE Integration

### `jaxkuramoto.odeint`
```python
def odeint(
    vector_fn: Callable,
    solver: Callable,
    t0: float,
    t1: float,
    dt: float,
    init_state: jnp.ndarray,
    observable_fn: Optional[Callable] = None
) -> Solution
```
Integrate ordinary differential equations using JAX-compiled `fori_loop`.

* **Parameters**:
  * `vector_fn`: Vector field $f(t, x)$.
  * `solver`: Stepper function (`euler` or `runge_kutta`).
  * `t0` (*float*): Initial time.
  * `t1` (*float*): Final time.
  * `dt` (*float*): Integration step size.
  * `init_state` (*jnp.ndarray*): Initial state at $t_0$.
  * `observable_fn` (*optional*): Function $g(t, x)$ to record observables at each time step. If `None`, records the full state.

* **Returns**:
  * `Solution`: Dataclass containing `ts`, `init_state`, `final_state`, and `observables`.

---

## Solvers (`jaxkuramoto.solver`)

* `euler(func, t, dt, state)`: Explicit 1st-order Euler method.
* `runge_kutta(func, t, dt, state)`: Classical 4th-order Runge-Kutta method (RK4).
* `integral_fn(func, a, minval, maxval, n)`: Trapezoidal numerical integration with custom VJP support.
* `fixed_point(func, a, x_guess, eps=1e-6)`: Fixed point solver using while-loop with reverse-mode automatic differentiation.

---

## Theory (`jaxkuramoto.theory`)

### `critical_point`
```python
def critical_point(pdf_fn: Callable, loc: float = 0.0) -> float
```
Calculate the theoretical critical coupling strength $K_c = \frac{2}{\pi g(0)}$ for symmetric unimodal natural frequency distribution $g(\omega)$.

### `orderparam`
```python
def orderparam(K: float, dist: Distribution, n=1000, r_guess=1.0, eps=1e-6) -> float
```
Solve the self-consistent equation for the steady-state order parameter $r$ given coupling strength $K$ and frequency distribution `dist`.

### `OttAntonsen`
```python
class OttAntonsen(ODE):
    def __init__(self, dist: Distribution, K: float) -> None
```
Ott-Antonsen ansatz reduction for infinite-oscillator ensembles with Lorentzian/Cauchy natural frequencies.

---

## Distributions (`jaxkuramoto.distribution`)

* `Normal(loc=0.0, scale=1.0)`: Gaussian distribution.
* `Cauchy(loc=0.0, gamma=1.0)`: Cauchy / Lorentzian distribution.
* `Uniform(low=0.0, high=1.0)`: Uniform distribution.
* `GeneralNormal(loc=0.0, gamma=1.0, n=1)`: Generalized normal distribution.
* `GeneralCauchy(loc=0.0, gamma=1.0, n=1)`: Generalized Cauchy distribution.
* `CauchyMultiply(gamma1, gamma2, Omega)`: Bimodal Cauchy mixture distribution.
* `FiniteDifferential(loc=0.0, scale=1.0, n=1)`: Compactly supported distribution.
