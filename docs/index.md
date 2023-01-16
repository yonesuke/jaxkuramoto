# Welcome to jaxkuramoto!

`jaxkuramoto` is a Python library for simulating the Kuramoto model, which has the following differential equation:

$$
\frac{\mathrm{d}\theta_{i}}{\mathrm{d} t}= \omega_{i} + \frac{K}{N}\sum_{j=1}^{N}\sin(\theta_{j}-\theta_{i}).
$$

It is built on top of [JAX](https://github.com/google/jax), a library for differentiable programming in Python. It is designed to be easy to use and to be compatible with JAX's JIT compilation and GPU acceleration.

![](figs/sync_nonsync.gif)

You can view the source code on [GitHub](https://github.com/yonesuke/jaxkuramoto).

```{tableofcontents}
```
