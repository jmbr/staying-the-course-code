from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from mayavi import mlab


DEFAULT_MESH_SIZE: int = 100
DEFAULT_AZIMUTH: int = 45
DEFAULT_ELEVATION: int = 90

DEFAULT_COLOR: Tuple[float] = (0.0, 0.0, 0.0)


def plot_spherical_potential(potential: Callable[[jnp.ndarray], float],
                             mesh_size: int = DEFAULT_MESH_SIZE,
                             azimuth: int = DEFAULT_AZIMUTH,
                             elevation: int = DEFAULT_ELEVATION) -> None:
    n = mesh_size
    u1, u2 = jnp.meshgrid(jnp.linspace(0.0, 2.0 * jnp.pi, n),
                          jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n))
    x1 = jnp.cos(u1) * jnp.cos(u2)
    x2 = jnp.sin(u1) * jnp.cos(u2)
    x3 = jnp.sin(u2)
    X = jnp.stack((x1.ravel(), x2.ravel(), x3.ravel())).T
    energy = jnp.clip(jax.vmap(potential)(X), -150.0, 20.0)
    mlab.mesh(x1, x2, x3, scalars=-energy.reshape(n, n), colormap='Blues')
    mlab.view(azimuth=azimuth, elevation=elevation)


def plot_fixed_points3d():
    from staying_the_course.potential import muller_fixed_points3d

    mlab.points3d(muller_fixed_points3d[::2, 0],
                  muller_fixed_points3d[::2, 1],
                  muller_fixed_points3d[::2, 2],
                  mode='sphere',
                  resolution=32,
                  scale_factor=0.025,
                  color=(1.0, 0.5, 0.0))
    mlab.points3d(muller_fixed_points3d[1::2, 0],
                  muller_fixed_points3d[1::2, 1],
                  muller_fixed_points3d[1::2, 2],
                  mode='cube',
                  resolution=32,
                  scale_factor=0.025,
                  color=(1.0, 0.5, 0.0))


def plot_points3d(points: jnp.ndarray, s=None, color=DEFAULT_COLOR) -> None:
    if s is not None:
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                      s, resolution=16, colormap='RdBu',
                      scale_mode='none', scale_factor=0.01)
    else:
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2],
                      color=color, resolution=16, scale_factor=0.01)


def plot_path3d(V, path3d, k=100):
    mlab.plot3d(path3d[:, 0], path3d[:, 1], path3d[:, 2],
                color=(0.0, 0.0, 0.0),
                tube_radius=0.0025, opacity=0.85)

    path_vectors3d = -jax.vmap(jax.jacobian(V))(path3d)
    path_vectors3d /= jnp.linalg.norm(path_vectors3d, axis=1)[..., None]

    s = slice(0, path3d.shape[0], k)
    q = mlab.quiver3d(path3d[s, 0], path3d[s, 1], path3d[s, 2],
                      path_vectors3d[s, 0],
                      path_vectors3d[s, 1],
                      path_vectors3d[s, 2],
                      color=(0.1, 0.1, 0.1),
                      mode='2ddash',
                      scale_mode='none', scale_factor=0.025)
    q.glyph.glyph_source.glyph_position = 'center'
    q.glyph.glyph_source.glyph_source.scale = 2.0
