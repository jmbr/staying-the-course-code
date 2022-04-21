from functools import partial

import jax
import jax.numpy as jnp

from .coordinates import Coordinates
from .diffusion_maps import DiffusionMaps
from .gaussian_process import GaussianProcess


class DiffusionMapCoordinates(Coordinates):
    def __init__(self, domain_dimension: int, codomain_dimension: int) -> None:
        self.domain_dimension = domain_dimension
        self.codomain_dimension = codomain_dimension
        self.diffusion_maps = DiffusionMaps(codomain_dimension + 1)
        self.gaussian_process = GaussianProcess()

    def learn(self, points: jnp.ndarray) -> None:
        self.diffusion_maps.learn(points)
        coordinates = (self.diffusion_maps.eigenvalues
                       * self.diffusion_maps.eigenvectors)
        self.gaussian_process._learn(points, coordinates,
                                     self.diffusion_maps.epsilon,
                                     self.diffusion_maps.kernel_matrix)
        return coordinates

    @partial(jax.jit, static_argnums=0)
    def __call__(self, point: jnp.ndarray) -> jnp.ndarray:
        return self.gaussian_process(point)
