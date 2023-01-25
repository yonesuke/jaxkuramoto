import jax.numpy as jnp

from jaxkuramoto.ode import ODE
from jaxkuramoto.distribution import Distribution

class OttAntonsen(ODE):
    """Ott-Antonsen reduction of the Kuramoto model."""
    def __init__(self, dist: Distribution, K: float) -> None:
        """Ott-Antonsen reduction of the Kuramoto model.

        Args:
            dist (Distribution): Distribution of natural frequencies.
            K (float): Coupling strength.
        """
        super().__init__()
        self.dist = dist
        self.dist_name = dist.__class__.__name__
        self.K = K
        if self.dist_name == "Cauchy":
            self.vector_fn = self._vector_fn_cauchy
            self.to_orderparam = lambda _, z: jnp.abs(z)
        elif self.dist_name == "CauchyMultiply":
            self.vector_fn = self._vector_fn_cauchymultiply
            self.k1 = dist.gamma2 * (2 * dist.Omega - 1j * (dist.gamma1 + dist.gamma2)) / (dist.gamma1 + dist.gamma2) / (2 * dist.Omega + 1j * (dist.gamma1 - dist.gamma2))
            self.k2 = dist.gamma1 * (2 * dist.Omega + 1j * (dist.gamma1 + dist.gamma2)) / (dist.gamma1 + dist.gamma2) / (2 * dist.Omega + 1j * (dist.gamma1 - dist.gamma2))
            self.zs2z = lambda zs: self.k1 * zs[0] + self.k2 * zs[1]
            self.to_orderparam = lambda _, zs: jnp.abs(self.zs2z(zs))
        else:
            raise ValueError("Distribution must be Cauchy or CauchyMultiply.")

    def _vector_fn_cauchy(self, t, z: jnp.ndarray) -> jnp.ndarray:
        """Vector field of Ott-Antonsen reduction of the Kuramoto model with the Cauchy distribution.

        Args:
            t (float): time
            z (jnp.ndarray): orderparameter.

        Returns:
            jnp.ndarray: Derivatives of orderparameter.
        """
        return 1j * self.dist.loc + (0.5 * self.K - self.dist.gamma) * z - 0.5 * self.K * jnp.abs(z)**2 * z

    def _vector_fn_cauchymultiply(self, t, zs: jnp.ndarray) -> jnp.ndarray:
        """Vector field of Ott-Antonsen reduction of the Kuramoto model with the CauchyMultiply distribution.

        Args:
            t (float): time
            zs (jnp.ndarray): Oscillator phases.

        Returns:
            jnp.ndarray: Derivatives of oscillator phases.
        """
        z1, z2 = zs
        z = self.zs2z(zs)
        dz1 = (1j * self.dist.Omega - self.dist.gamma1) * z1 - 0.5 * self.K * (z.conj() * z1 * z1 - z)
        dz2 = -(1j * self.dist.Omega + self.dist.gamma2) * z2 - 0.5 * self.K * (z.conj() * z2 * z2 - z)
        return jnp.array([dz1, dz2])