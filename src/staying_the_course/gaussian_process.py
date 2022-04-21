from typing import Optional

from functools import partial

import scipy.spatial.distance as distance
import jax
import jax.scipy
import jax.numpy as jnp


DEFAULT_SIGMA: float = 1e-4


class GaussianProcess:
    """Gaussian process regressor."""
    sigma: float = DEFAULT_SIGMA         # Regularization term.
    epsilon: Optional[float] = None      # Spatial scale parameter.
    alphas: Optional[jnp.ndarray] = None  # Coefficients.

    def __init__(self, sigma: Optional[float] = None,
                 epsilon: Optional[float] = None) -> None:
        if sigma is not None:
            self.sigma = sigma
        if epsilon is not None:
            self.epsilon = epsilon

    def _learn(self, points: jnp.ndarray, values: jnp.ndarray,
               epsilon: float, kernel_matrix: jnp.ndarray) -> None:
        """Auxiliary method for fitting a Gaussian process."""
        self.points = points
        self.epsilon = epsilon

        sigma2_eye = self.sigma**2 * jnp.eye(kernel_matrix.shape[0])
        L, _ = jax.scipy.linalg.cho_factor(
            kernel_matrix + sigma2_eye, lower=True, check_finite=False)
        self.cholesky_factor = L
        self.alphas = jax.scipy.linalg.cho_solve((L, True), values,
                                                 check_finite=False)

    def learn(self, points: jnp.ndarray, values: jnp.ndarray) -> None:
        """Fit a Gaussian process

        Parameters
        ----------
        points: jnp.ndarray
            Data points arranged by rows.
        values: jnp.ndarray
            Values corresponding to the data points. These can be scalars or
            arrays (arranged by rows).

        """
        distances2 = distance.pdist(points, metric='sqeuclidean')

        if self.epsilon is None:
            threshold = jnp.finfo(points.dtype).eps * 1e2
            self.epsilon = jnp.sqrt(
                jnp.median(distances2[distances2 > threshold]))

        kernel_matrix = distance.squareform(
            jnp.exp(-distances2 / (2.0 * self.epsilon**2)))
        diagonal_indices = jnp.diag_indices_from(kernel_matrix)
        kernel_matrix[diagonal_indices] = 1.0
        self.kernel_matrix = kernel_matrix

        self._learn(points, values,
                    self.epsilon, self.kernel_matrix)

    @partial(jax.jit, static_argnums=0)
    def __call__(self, point: jnp.ndarray) -> jnp.ndarray:
        """Evaluate Gaussian process at a new point.

        This function must be called after the Gaussian process has been
        fitted using the `learn` method.

        Parameters
        ----------
        point: jnp.ndarray
            A single point on which the previously learned Gaussian process
            is to be evaluated.

        Returns
        -------
        value: jnp.ndarray
            Estimated value of the GP at the given point.

        """
        kstar = jnp.exp(
            -jnp.sum((point - self.points)**2, axis=1)
            / (2.0 * self.epsilon**2))
        return kstar @ self.alphas


def make_gaussian_process(X: jnp.ndarray, Y: jnp.ndarray) -> GaussianProcess:
    X, Y = jnp.atleast_2d(X), jnp.atleast_2d(Y)
    assert X.shape[0] == Y.shape[0]
    f = GaussianProcess()
    f.learn(X, Y)
    return f
