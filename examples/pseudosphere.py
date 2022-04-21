from IPython import get_ipython
get_ipython().run_line_magic('reset', '-fs')

from typing import Callable

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from staying_the_course.potential import potential2d, muller_fixed_points
from staying_the_course.isocline_field import IsoclineField, compute_path


random_seed: int = 0
key = jax.random.PRNGKey(random_seed)
np.random.seed(random_seed)


dt: float = 5e-4
num_steps: int = 50000


@jax.jit
def psi(x):
    return jnp.array([jnp.linalg.norm(x[:2]),
                      jnp.arctan2(x[0], x[1])])


@jax.jit
def phi(u):
    return jnp.array([u[0] * jnp.cos(u[1]),
                      u[0] * jnp.sin(u[1]),
                      jnp.arccosh(1.0 / u[0])
                      - jnp.sqrt(1.0 - u[0]**2)])


@jax.jit
def polar_to_cartesian(u):
    return jnp.array([
        0.9867606472 * u[1] - 1.85, 4.406507321 * u[0] - 1.715856588
    ])


@jax.jit
def cartesian_to_polar(x):
    return jnp.array([
        0.3893915211 + 0.2269371017 * x[1], 1.013416985 * x[0] + 1.874821422
    ])


def make_points(u1min, u1max, n1, u2min, u2max, n2):
    u1, u2 = jnp.meshgrid(
        jnp.linspace(u1min, u1max, n1),
        jnp.linspace(u2min, u2max, n2))
    return jnp.stack((u1.ravel(), u2.ravel())).T


@jax.jit
def potential(u):
    return potential2d(polar_to_cartesian(u))


@jax.jit
def potential3d(x):
    return potential(polar_to_cartesian(psi(x)))


@jax.jit
def vector_field(u):
    return -jax.grad(potential)(u)


@jax.jit
def normalized_vector_field(u):
    vector = vector_field(u)
    return vector / jnp.linalg.norm(vector)


muller_fixed_points_2d = jax.vmap(cartesian_to_polar)(muller_fixed_points)
sinks = muller_fixed_points_2d[::2, :]
saddles = muller_fixed_points_2d[1::2, :]


def plot_potential(V, n=100, sinks=None, saddles=None, ax=None,
                   colorbar=True):
    if ax is None:
        ax = plt.gca()

    u1, u2 = jnp.meshgrid(jnp.linspace(0.2759229703, 1.0, n),
                          jnp.linspace(0.0, jnp.pi, n))
    U = jnp.stack((u1.ravel(), u2.ravel())).T
    energies = jax.vmap(potential)(U).reshape(n, n)

    c = ax.pcolormesh(u1, u2, energies, vmin=-150, vmax=20, cmap='Blues_r',
                      rasterized=True)
    if colorbar is True:
        plt.colorbar(c, label='Energy')

    if sinks is not None:
        ax.scatter(sinks[:, 0], sinks[:, 1],
                   marker='o', c='orange', s=20, zorder=1000)
    if saddles is not None:
        ax.scatter(saddles[:, 0], saddles[:, 1],
                   marker='s', c='orange', s=20, zorder=1000)

    ax.set_xticks(jnp.linspace(0.2759229703, 1.0, 5))
    ax.set_yticks([0.0, jnp.pi / 4.0, jnp.pi / 2.0, 3.0 * jnp.pi / 4.0, jnp.pi])
    ax.set_xlim(0.2759229703, 1.0)
    ax.set_ylim(0.0, jnp.pi)
    ax.set_xlabel(r'$u^1$')
    ax.set_ylabel(r'$u^2$')
    aspect_ratio = (1.0 - 0.2759229703) / (jnp.pi)
    ax.set_aspect(aspect_ratio)


def plot_vector_field(vector_field: Callable, n1, n2, ax=None, L=1.75,
                      *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    p1, p2 = jnp.meshgrid(jnp.linspace(0.2759229703, 1.0, n1),
                          jnp.linspace(0.0, jnp.pi, n2))
    P = jnp.stack((p1.ravel(), p2.ravel())).T

    W = jax.vmap(vector_field)(P)

    ax.quiver(P[:, 0], P[:, 1], W[:, 0], W[:, 1],
              pivot='middle', headwidth=0, headlength=0, headaxislength=0,
              *args, **kwargs)


U = make_points(u1min=0.2759229703, u1max=1.0, n1=50,
                u2min=0.0, u2max=2.0 * jnp.pi, n2=50)

isocline_field = IsoclineField(phi, psi, vector_field)

u0 = jnp.array([0.76453468, 1.17568036])
direction = jnp.array([1.0, 0.0])
path1 = compute_path(isocline_field, u0, direction,
                     num_steps=num_steps, max_step=dt)
path2 = compute_path(isocline_field, u0, -direction,
                     num_steps=num_steps, max_step=dt)

fig = plt.figure()
plot_potential(potential, n=100, sinks=sinks, saddles=saddles)
plot_vector_field(normalized_vector_field, n1=30, n2=30)
plt.plot(path1[:, 0], path1[:, 1], c='k', lw=2, alpha=0.75)
plt.plot(path2[:, 0], path2[:, 1], c='k', lw=2, alpha=0.75)
plt.show()
