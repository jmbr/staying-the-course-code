import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns


def cartesian_to_polar(x):
    return jnp.array([jnp.arctan2(x[1], x[0]),
                      jnp.arctan2(x[2], jnp.sqrt(x[0]**2 + x[1]**2))])


def polar_to_cartesian(u):
    x1 = jnp.cos(u[0]) * jnp.cos(u[1])
    x2 = jnp.sin(u[0]) * jnp.cos(u[1])
    x3 = jnp.sin(u[1])
    return jnp.array([x1, x2, x3])


def plot_polar_potential(V, n=100, sinks=None, saddles=None, ax=None,
                         colorbar=True):
    if ax is None:
        ax = plt.gca()

    u1, u2 = jnp.meshgrid(jnp.linspace(0, jnp.pi, n),
                          jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n))
    # U = jnp.stack((u1.ravel(), u2.ravel())).T
    x1 = jnp.cos(u1) * jnp.cos(u2)
    x2 = jnp.sin(u1) * jnp.cos(u2)
    x3 = jnp.sin(u2)
    X = jnp.stack((x1.ravel(), x2.ravel(), x3.ravel())).T
    energies = jax.vmap(V)(X).reshape(n, n)

    sns.set_style('ticks')
    c = ax.pcolormesh(u1, u2, energies, vmin=-150, vmax=20, cmap='Blues_r',
                      rasterized=True)
    if colorbar is True:
        cb = plt.colorbar(c)

    if sinks is not None:
        ax.scatter(sinks[:, 0], sinks[:, 1],
                   marker='o', c='orange', s=20, zorder=1000)
    if saddles is not None:
        ax.scatter(saddles[:, 0], saddles[:, 1],
                   marker='s', c='orange', s=20, zorder=1000)

    ax.set_xticks([0, jnp.pi / 4, jnp.pi / 2],
                  labels=['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
    ax.set_yticks([-jnp.pi / 2, -jnp.pi / 4, 0, jnp.pi / 4, jnp.pi / 2],
                  labels=[r'$-\frac{\pi}{2}$',
                          r'$-\frac{\pi}{4}$',
                          '0',
                          r'$\frac{\pi}{4}$',
                          r'$\frac{\pi}{2}$'])
    ax.set_xlim(0.0, jnp.pi / 2)
    ax.set_ylim(-jnp.pi / 4, jnp.pi / 4)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    if colorbar is True:
        cb.set_label('Potential energy')
    ax.set_aspect(1.0)
    sns.despine()


def plot_path2d(path3d, color, ax=None):
    if ax is None:
        ax = plt.gca()
    path = jax.vmap(cartesian_to_polar)(path3d)

    sns.set_style('ticks')
    ax.plot(path[:, 0], path[:, 1], c=color, alpha=0.75,
            zorder=10000, rasterized=True)
    sns.despine()


def plot_points_and_vectors(points, vectors,  # colors,
                            distinguished_point=None, ax=None):
    if ax is None:
        ax = plt.gca()
    # ax.scatter(points[:, 0], points[:, 1], s=10, cmap='RdBu_r')
    ax.quiver(points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1],
              # colors, cmap='RdBu',
              pivot='middle', headlength=0, headwidth=0, headaxislength=0)
    if distinguished_point is not None:
        ax.scatter([distinguished_point[0]], [distinguished_point[1]],
                   s=10, c='red')
