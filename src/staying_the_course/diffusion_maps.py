from typing import Optional

import numpy as np
import scipy.spatial
import jax.numpy as jnp


class DiffusionMaps:
    epsilon: Optional[float] = None
    num_eigenpairs: int
    points: jnp.ndarray
    kernel_matrix: jnp.ndarray
    eigenvalues: jnp.ndarray
    eigenvectors: jnp.ndarray

    def __init__(self, num_eigenpairs: int,
                 epsilon: Optional[float] = None) -> None:
        if epsilon is not None:
            self.epsilon = epsilon
        self.num_eigenpairs = num_eigenpairs

    def learn(self, points: jnp.ndarray) -> None:
        distances2 = scipy.spatial.distance.pdist(
            points, metric='sqeuclidean')

        if self.epsilon is None:  # Guess spatial scale.
            threshold = jnp.finfo(points.dtype).eps * 1e2
            self.epsilon = jnp.sqrt(
                jnp.median(distances2[distances2 > threshold]))

        kernel_matrix = scipy.spatial.distance.squareform(
            np.exp(-distances2 / (2.0 * self.epsilon**2)))
        kernel_matrix[np.diag_indices_from(kernel_matrix)] = 1.0
        self.kernel_matrix = kernel_matrix

        inv_sqrt_diag_vector = 1.0 / np.sqrt(np.sum(kernel_matrix, axis=0))
        normalized_kernel_matrix = ((kernel_matrix * inv_sqrt_diag_vector).T
                                    * inv_sqrt_diag_vector).T

        ew, ev = scipy.sparse.linalg.eigsh(normalized_kernel_matrix,
                                           k=self.num_eigenpairs,
                                           v0=np.ones(kernel_matrix.shape[0]))
        indices = np.argsort(np.abs(ew))[::-1]
        ew = ew[indices]
        ev = ev[:, indices] * inv_sqrt_diag_vector[:, None]
        self.eigenvalues = ew[1:]
        self.eigenvectors = ev[:, 1:]

        # assert np.allclose((np.diag(1.0 / kernel_matrix.sum(axis=1))
        #                     @ kernel_matrix @ ev),
        #                    ev @ np.diag(ew))  # Assertion only for testing.
