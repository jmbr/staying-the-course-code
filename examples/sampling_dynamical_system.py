from IPython import get_ipython
get_ipython().run_line_magic('reset', '-fs')

from typing import Optional

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
plt.close()
from mayavi import mlab

from staying_the_course.potential import potential3d as V, muller_fixed_points3d
from staying_the_course.metropolis import Metropolis
from staying_the_course.diffusion_map_coordinates import DiffusionMapCoordinates
from staying_the_course.isocline_field import IsoclineField, compute_path
from staying_the_course.gaussian_process import make_gaussian_process

from plot3d import (plot_spherical_potential, plot_points3d,
                    plot_fixed_points3d, plot_path3d)
from plot2d import (plot_polar_potential, plot_path2d, cartesian_to_polar,
                    plot_points_and_vectors)


# Set up random number generator.
random_seed: int = 0
key = jax.random.PRNGKey(random_seed)
np.random.seed(random_seed)

fixed_points_polar = jax.vmap(cartesian_to_polar)(muller_fixed_points3d)
sinks, saddles = fixed_points_polar[::2, :], fixed_points_polar[1::2, :]

temperature: float = 1e0
delta: float = 1e-3

initial_point: np.ndarray = (muller_fixed_points3d[0, :]
                             + 1e-5 * jax.random.normal(key, (3,)))

direction = jnp.array([-1.0, 0.0])  # Preferred direction.


class SphericalMetropolis(Metropolis):
    def _generate_candidate(self):
        x = self.current_point
        u1 = np.arctan2(x[1], x[0])
        u2 = np.arctan2(x[2], np.sqrt(x[0]**2 + x[1]**2))
        v = (np.array([1.0 / np.cos(u2), 1.0])
             * np.random.randn(2) * self.delta)
        Df = np.array([[-np.sin(u1) * np.cos(u2),
                        -np.cos(u1) * np.sin(u2)],
                       [np.cos(u1) * np.cos(u2),
                        -np.sin(u1) * np.sin(u2)],
                       [0, np.cos(u2)]])
        xp = x + Df @ v
        return xp / np.linalg.norm(xp)

    def sample_batches(self, num_batches: int,
                       num_samples_per_batch: int) -> jnp.ndarray:
        X = None

        for k in range(num_batches):
            self.reset()

            samples = self.draw_samples(num_samples_per_batch)

            if X is None:
                X = samples.copy()
            else:
                X = np.vstack((X, samples))

        return X


def make_vector_field(coordinates, points):
    forces = -jax.vmap(jax.jacobian(V))(points)
    mapped_points = jax.vmap(coordinates)(points)
    mapped_forces = jnp.einsum('...ij,...j',
                               jax.vmap(jax.jacobian(coordinates))(points),
                               forces)
    return make_gaussian_process(mapped_points, mapped_forces)


figure, ax = plt.subplots(1, 2)
paths3d = None
for i in range(200):
    print(f'Iteration #{i}')

    sampler = SphericalMetropolis(
        V, temperature=temperature, delta=delta, initial_point=initial_point)
    X = sampler.sample_batches(num_batches=100, num_samples_per_batch=10)

    coordinates = DiffusionMapCoordinates(X.shape[1], 2)
    Y = coordinates.learn(X)

    psi = coordinates
    phi = make_gaussian_process(Y, X)
    vector_field = make_vector_field(psi, X)
    isocline_field = IsoclineField(phi, psi, vector_field)

    initial_point_coordinates = coordinates(initial_point)
    if paths3d is not None:
        mapped_points = jax.vmap(coordinates)(paths3d[-2:, :])
        direction = jnp.diff(mapped_points, axis=0).squeeze()
    path = compute_path(isocline_field,
                        initial_point_coordinates, direction,
                        num_steps=100, max_step=5e-6)
    path3d = jax.vmap(phi)(path)
    if paths3d is None:
        paths3d = path3d.copy()
    else:
        paths3d = jnp.concatenate((paths3d, path3d))

    initial_point = paths3d[-1, :].copy()

    VY = jnp.einsum('...ij,...j',
                    jax.vmap(jax.jacobian(coordinates))(X),
                    -jax.vmap(jax.jacobian(V))(X))
    VY /= jnp.linalg.norm(VY, axis=1)[..., None]

    ax[0].clear()
    ax[1].clear()

    plot_polar_potential(V, n=150, ax=ax[0], colorbar=False)
    plot_path2d(paths3d, ax=ax[0], color='black')

    plot_points_and_vectors(Y, VY, ax=ax[1],
                            distinguished_point=initial_point_coordinates)
    ax[1].plot(path[:, 0], path[:, 1], c='r', alpha=0.9, lw=5)

    plt.pause(0.01)
    plt.tight_layout()
