from typing import Callable
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


class IsoclineField:
    def __init__(self, phi: Callable[[jnp.ndarray], jnp.ndarray],
                 psi: Callable[[jnp.ndarray], jnp.ndarray],
                 vector_field: Callable[[jnp.ndarray], jnp.ndarray]) -> None:
        self.phi = phi          # Parameterization
        self.Dphi = jax.jacobian(phi)

        self.psi = psi          # System of coordinates (psi ∘ phi = id))
        self.Dpsi = jax.jacobian(psi)

        self._original_vector_field = vector_field

    @partial(jax.jit, static_argnums=0)
    def _metric_tensor(self, u: jnp.ndarray) -> jnp.ndarray:
        """Return the Riemannian metric tensor at a point."""
        Dphi = self.Dphi(u)
        return Dphi.T @ Dphi

    @partial(jax.jit, static_argnums=0)
    def _inverse_metric_tensor(self, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.inv(self._metric_tensor(u))

    @partial(jax.jit, static_argnums=0)
    def _christoffel(self, u: jnp.ndarray) -> jnp.ndarray:
        g_inv = self._inverse_metric_tensor(u)
        Dg = jax.jacobian(self._metric_tensor)(u)
        return 0.5 * (- jnp.einsum('ijl,lk', Dg, g_inv)
                      + jnp.einsum('lij,lk', Dg, g_inv)
                      + jnp.einsum('jli,lk', Dg, g_inv))

    @partial(jax.jit, static_argnums=0)
    def _vector_field(self, u):
        g_u = self._metric_tensor(u)
        x = self._original_vector_field(u)
        return x / jnp.sqrt(jnp.dot(x, g_u @ x))

    @partial(jax.jit, static_argnums=0)
    def _generate_matrix(self, u):
        """Returns matrix representation of parallel transport equations at a point for the vector as its unknown."""
        return (jax.jacobian(self._vector_field)(u).T
                + jnp.einsum(
                    'ijk,j', self._christoffel(u), self._vector_field(u)))

    @partial(jax.jit, static_argnums=0)
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        U, s, V = jnp.linalg.svd(self._generate_matrix(u))
        # assert s[-1] < 1e-12
        return U[-1, :]


def compute_path(isocline_field, initial_condition, direction,
                 num_steps, max_step):
    """Compute continuation path of isocline field.

    """
    h = max_step

    vprev = direction

    ps = np.zeros((num_steps, initial_condition.shape[0]))
    ps[0, :] = initial_condition
    for i in range(1, num_steps):
        p = ps[i - 1, :]
        v = isocline_field(p)
        if vprev is not None:
            v *= jnp.sign(jnp.dot(v, vprev))
        vprev = v
        quadratic_correction = jnp.einsum(
            'i,ijk,j', v, isocline_field._christoffel(ps[i - 1, :]), v).sum()
        ps[i, :] = ps[i - 1, :] + h * v - h**2 * quadratic_correction

    return ps
