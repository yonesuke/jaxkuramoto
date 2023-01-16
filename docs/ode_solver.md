# List of available ODE solvers

## Euler method

```{prf:algorithm} Euler method
:label: euler

**Inputs** A function $f$ and an initial condition $y_0$

**Output** A sequence of approximations $y_n$ to the solution $y(t)$

1. Set $y_0 = y(0)$
2. For $n = 0, 1, 2, \ldots$ do
    1. Set $y_{n+1} = y_n + hf(t_n, y_n)$
```

## Runge–Kutta method

```{prf:algorithm} Runge–Kutta method
:label: runge-kutta

**Inputs** A function $f$ and an initial condition $y_0$

**Output** A sequence of approximations $y_n$ to the solution $y(t)$

1. Set $y_0 = y(0)$
2. For $n = 0, 1, 2, \ldots$ do
    1. Set $k_1 = hf(t_n, y_n)$
    2. Set $k_2 = hf(t_n + \frac{h}{2}, y_n + \frac{k_1}{2})$
    3. Set $k_3 = hf(t_n + \frac{h}{2}, y_n + \frac{k_2}{2})$
    4. Set $k_4 = hf(t_n + h, y_n + k_3)$
    5. Set $y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$
```