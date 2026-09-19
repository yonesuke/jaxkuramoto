**********************
Package Reference
**********************

Kuramoto model
#################################

.. autoclass:: jaxkuramoto.Kuramoto
    :members:

.. autoclass:: jaxkuramoto.SakaguchiKuramoto
    :members:

Solver
#################################

ODE solver
*********************************

.. autofunction:: jaxkuramoto.solver.euler
.. autofunction:: jaxkuramoto.solver.runge_kutta

Integral
*********************************
.. autofunction:: jaxkuramoto.solver.integral_fn

Fixed point solver
*********************************

.. autofunction:: jaxkuramoto.solver.fixed_point


ODE Integration
#################################

.. autofunction:: jaxkuramoto.odeint


Solution
#################################

.. autoclass:: jaxkuramoto.Solution
    :members:

Theory
#################################

.. autofunction:: jaxkuramoto.theory.orderparam

.. autofunction:: jaxkuramoto.theory.critical_point

.. autoclass:: jaxkuramoto.theory.OttAntonsen
    :members:

Distribution
#################################

.. autoclass:: jaxkuramoto.distribution.Distribution
    :members:

.. autoclass:: jaxkuramoto.distribution.Normal
    :members:

.. autoclass:: jaxkuramoto.distribution.Cauchy
    :members:

.. autoclass:: jaxkuramoto.distribution.Uniform
    :members:

.. autoclass:: jaxkuramoto.distribution.GeneralNormal
    :members:

.. autoclass:: jaxkuramoto.distribution.GeneralCauchy
    :members:

.. autoclass:: jaxkuramoto.distribution.CauchyMultiply
    :members:

.. autoclass:: jaxkuramoto.distribution.FiniteDifferential
    :members: