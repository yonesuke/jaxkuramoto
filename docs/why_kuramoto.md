# Why Kuramoto model?

The **Kuramoto model** is a mathematical model used to describe the synchronization behavior of a large set of coupled oscillators. Proposed by Yoshiki Kuramoto in 1975, it has become one of the most celebrated and ubiquitous paradigms in nonlinear dynamics and statistical physics.

## Synchronization phenomena

Synchronization is observed across diverse systems in nature, biology, and engineering:

- **Biological systems**: Synchronous flashing of Southeast Asian fireflies, cardiac pacemaker cells firing in unison, and collective neural oscillations in the brain.
- **Physics and engineering**: Alternating current (AC) power grid networks, arrays of superconducting Josephson junctions, and coupled lasers.
- **Social dynamics**: Rhythmic clapping in concert halls and pedestrian-induced swaying of suspension bridges (such as the Millennium Bridge).

The standard governing equation for $N$ phase oscillators is given by:

$$
\frac{\mathrm{d}\theta_{i}}{\mathrm{d} t} = \omega_{i} + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_{j} - \theta_{i}), \quad i = 1, \ldots, N
$$

where $\theta_i$ represents the phase of oscillator $i$, $\omega_i$ is its intrinsic natural frequency, and $K$ is the coupling strength.

## Why `jaxkuramoto`?

While many implementations of the Kuramoto model exist, `jaxkuramoto` is built from the ground up using [Google JAX](https://github.com/google/jax). This provides several unique advantages:

### 1. Massive GPU acceleration
Evaluating the all-to-all interactions scales as $\mathcal{O}(N^2)$ (or $\mathcal{O}(N)$ using order parameter formulations). With JAX's accelerated linear algebra (XLA) backend, simulating systems with tens or hundreds of thousands of oscillators can be executed in parallel on GPUs/TPUs with orders-of-magnitude speedups compared to traditional CPU implementations.

### 2. Differentiable dynamical systems
Because JAX supports automatic differentiation (`jax.grad`, `jax.vjp`), `jaxkuramoto` turns Kuramoto simulations into differentiable computational graphs. This opens up possibilities for:
- Learning interaction topologies or coupling parameters from observation data via gradient descent.
- Sensitivity analysis and Lyapunov exponent calculations.
- Seamless integration with modern machine learning frameworks (e.g., neural ODEs).

### 3. Vectorization (`vmap`) and JIT compilation
With `jax.jit`, ODE stepping loops run with near-zero Python overhead. Furthermore, `jax.vmap` allows batching multiple parameter settings (such as varying coupling strengths $K$ or initial distributions) with single-line syntax.

### 4. Integration with theoretical reductions
In addition to direct numerical simulation, `jaxkuramoto` provides built-in tools for analytical theory, including the **Ott-Antonsen ansatz** and self-consistent order parameter calculations for various frequency distributions.