import jax.numpy as jnp

from .ode import ODE

class Kuramoto(ODE):
    """Kuramoto model."""
    def __init__(self, omegas: jnp.ndarray, K: float) -> None:
        """Kuramoto model.

        Args:
            omegas (jnp.ndarray): Natural frequencies of oscillators.
            K (float): Coupling strength.
        """
        self.omegas = omegas
        self.K = K
        self.n_oscillator = omegas.shape[0]

    def vector_fn(self, t, thetas: jnp.ndarray) -> jnp.ndarray:
        """Vector field of Kuramoto model.
        
        Args:
            t (float): time
            thetas (jnp.ndarray): Oscillator phases.
        
        Returns:
            jnp.ndarray: Derivatives of oscillator phases.
        """
        coss, sins = jnp.cos(thetas), jnp.sin(thetas)
        rx, ry = coss.mean(), sins.mean()
        return self.omegas + self.K * (ry * coss - rx * sins)

    def orderparameter(self, thetas: jnp.ndarray) -> float:
        """Order parameter.

        Args:
            thetas (jnp.ndarray): Oscillator phases.
            
        Returns:
            float: Order parameter.    
        """
        rx, ry = jnp.cos(thetas).mean(), jnp.sin(thetas).mean()
        return jnp.sqrt(rx * rx + ry * ry)


class SakaguchiKuramoto(ODE):
    """Sakaguchi-Kuramoto model."""
    def __init__(self, omegas: jnp.ndarray, K: float, alpha: float) -> None:
        """Sakaguchi-Kuramoto model.

        Args:
            omegas (jnp.ndarray): Natural frequencies of oscillators.
            K (float): Coupling strength.
            alpha (float): Phase shift.
        """
        super().__init__()
        self.omegas = omegas
        self.K = K
        self.alpha = alpha
        self.n_oscillator = omegas.shape[0]

    def vector_fn(self, t, thetas: jnp.ndarray) -> jnp.ndarray:
        """Vector field of Kuramoto-Sakaguchi model.
        
        Args:
            t (float): time
            thetas (jnp.ndarray): Oscillator phases.
        
        Returns:
            jnp.ndarray: Derivatives of oscillator phases.
        """
        coss, sins = jnp.cos(thetas + self.alpha), jnp.sin(thetas + self.alpha)
        rx, ry = coss.mean(), sins.mean()
        return self.omegas + self.K * (ry * coss - rx * sins)

    def orderparameter(self, thetas: jnp.ndarray) -> float:
        """Order parameter.

        Args:
            thetas (jnp.ndarray): Oscillator phases.
            
        Returns:
            float: Order parameter.    
        """
        rx, ry = jnp.cos(thetas).mean(), jnp.sin(thetas).mean()
        return jnp.sqrt(rx * rx + ry * ry)

class NetworkKuramoto(ODE):
    """Network Kuramoto model."""
    def __init__(self, kuramoto, adjacency_matrix):
        """Network Kuramoto model.

        Args:
            kuramoto (Kuramoto): Kuramoto model.
            adjacency_matrix (jnp.ndarray): Adjacency matrix of network.
        """
        self.kuramoto = kuramoto
        self.A = adjacency_matrix

    def vector_fn(self, t, thetas):
        coss, sins = jnp.cos(thetas), jnp.sin(thetas)
        rx, ry = coss.mean(), sins.mean()
        pass

class CirculantKuramoto(ODE):
    def __init__(self, kuramoto, xs):
        self.kuramoto = kuramoto
        self.xs = xs

    def vector_fn(self, t, thetas):
        pass

    