# List of available nonlinear solver

## Iterative method

```{prf:algorithm} Iterative method
:label: iterative

**Inputs** A function $f$ and an initial guess $x_0$

**Output** An approximation $x$ to a root of $f$

1. Set $x_0$
2. For $n = 0, 1, 2, \ldots$ do
    1. Set $x_{n+1} = f(x_n)$
```

## Newton's method

```{prf:algorithm} Newton's method
:label: newton

**Inputs** A function $f$ and an initial guess $x_0$

**Output** An approximation $x$ to a root of $f$

1. Set $x_0$
2. For $n = 0, 1, 2, \ldots$ do
    1. Set $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$
```