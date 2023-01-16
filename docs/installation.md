# Installation

## Stable version

The latest stable release of `jaxkuramoto` can be installed via `pip`:

```bash
pip install jaxkuramoto
```

## GPU support

Kuramoto model is a perfect candidate for GPU acceleration since the model is highly parallelizable.

If you want to run the `jaxkuramoto` code in a GPU environment,
you will first need to install the `jax` with compatible GPU version.
Check out the [JAX installation guide](https://github.com/google/jax#pip-installation-gpu-cuda)
for more details.

After installing the `jax` with GPU support, you can install `jaxkuramoto` simply by running the `pip` command above.
