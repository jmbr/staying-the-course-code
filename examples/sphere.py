from IPython import get_ipython
get_ipython().run_line_magic('reset', '-fs')

from typing import Callable

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from staying_the_course.potential import potential3d
from staying_the_course.isocline_field import IsoclineField, compute_path


random_seed: int = 0
key = jax.random.PRNGKey(random_seed)
np.random.seed(random_seed)


dt: float = 1e-4
num_steps: int = 10000


@jax.jit
def psi(x):
    """Stereographic projection (from North pole)."""
    return x[:2] / (1.0 - x[2])


@jax.jit
def phi(u):
    """Inverse of stereographic projection."""
    u2 = jnp.dot(u, u)
    return jnp.array([2.0 * u[0], 2.0 * u[1], u2 - 1.0]) / (u2 + 1.0)


@jax.jit
def potential(u):
    return potential3d(phi(u))


@jax.jit
def vector_field(u):
    return -jax.grad(potential)(u)


def plot_energy(energy: Callable, ax=None, n=200, L=1.75, *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    p1, p2 = jnp.meshgrid(jnp.linspace(0.0, L, n), jnp.linspace(0.0, L, n))
    P = jnp.stack((p1.ravel(), p2.ravel())).T

    energies = jnp.clip(jax.vmap(energy)(P), -150.0, 20.0)

    pc = ax.pcolormesh(p1, p2, energies.reshape(n, n),
                       cmap='Blues_r', rasterized=True)
    plt.colorbar(pc, label='Energy', ax=ax)


def plot_vector_field(vector_field: Callable, ax=None, n=30, L=1.75,
                      *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    p1, p2 = jnp.meshgrid(jnp.linspace(0.0, L, n), jnp.linspace(0.0, L, n))
    P = jnp.stack((p1.ravel(), p2.ravel())).T

    W = jax.vmap(vector_field)(P)

    ax.quiver(P[:, 0], P[:, 1], W[:, 0], W[:, 1],
              pivot='middle', headwidth=0, headlength=0, headaxislength=0,
              *args, **kwargs)
    ax.set_xlim(0.0, L)
    ax.set_ylim(0.0, L)
    ax.set_aspect('equal')


isocline_field = IsoclineField(phi, psi, vector_field)


def onclick(event):
    ax = event.inaxes
    initial_point = jnp.array([event.xdata, event.ydata])

    path1 = compute_path(isocline_field, initial_point,
                         jnp.array([1.0, 0.0]),
                         num_steps=num_steps, max_step=dt)
    ax.plot(path1[:, 0], path1[:, 1], c='k', lw=2)

    path2 = compute_path(isocline_field,
                         initial_point, jnp.array([-1.0, 0.0]),
                         num_steps=num_steps, max_step=dt)
    ax.plot(path2[:, 0], path2[:, 1], c='k', lw=2)
    fig.canvas.draw_idle()


print('Click on the graph to compute isoclines passing through a point.')

fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plot_energy(potential)
muller_fixed_points3d = jnp.array(
    [[0.27632632, 0.84100892, -0.46513199],
     [0.47435655, 0.81760542, -0.32635446],
     [0.59553828, 0.76936331, -0.23111522],
     [0.85849672, 0.49256384, -0.14270336],
     [0.75211518, 0.5771847, 0.31808894]])
muller_fixed_points = jax.vmap(psi)(muller_fixed_points3d)
plt.scatter(muller_fixed_points[::2, 0], muller_fixed_points[::2, 1], s=5,
            marker='o', color='orange', zorder=1000)
plt.scatter(muller_fixed_points[1::2, 0], muller_fixed_points[1::2, 1], s=5,
            marker='s', color='orange', zorder=1000)
plt.xlim(0, 1.75)
plt.ylim(0, 1.75)
plt.xlabel('$p^1$')
plt.ylabel('$p^2$')
plt.show()
