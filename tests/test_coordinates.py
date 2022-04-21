import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from staying_the_course.coordinates import Coordinates


def test_identity_coordinates():
    class IdentityCoordinates(Coordinates):
        def __init__(self, domain_dimension, codomain_dimension):
            self.domain_dimension = domain_dimension
            self.codomain_dimension = codomain_dimension

        def __call__(self, X):
            return X

    identity_coordinates = IdentityCoordinates(3, 3)
    X = jnp.arange(3 * 10, dtype=jnp.float64).reshape(10, 3)
    Y = identity_coordinates(X)
    assert jnp.allclose(X, Y)


def test_spherical_coordinates():
    class SphericalCoordinates(Coordinates):
        domain_dimension = 3
        codomain_dimension = 2

        def __call__(self, x):
            assert x.shape[0] == self.domain_dimension
            u1 = jnp.arctan2(x[1], x[0])
            u2 = jnp.arctan2(x[2], jnp.sqrt(x[0]**2 + x[1]**2))
            return jnp.array([u1, u2])

    spherical_coordinates = SphericalCoordinates()

    n1 = n2 = 100
    u1, u2 = jnp.meshgrid(jnp.linspace(0.0, 2.0 * jnp.pi, n1, endpoint=False),
                          jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, n2,
                                       endpoint=False))

    x1 = jnp.cos(u1) * jnp.cos(u2)
    x2 = jnp.sin(u1) * jnp.cos(u2)
    x3 = jnp.sin(u2)
    X = jnp.stack((x1.ravel(), x2.ravel(), x3.ravel())).T

    U = jax.vmap(spherical_coordinates)(X)

    tol = jnp.finfo(X.dtype).eps * 1e1
    assert jnp.allclose(X[:, 0], jnp.cos(U[:, 0]) * jnp.cos(U[:, 1]), atol=tol)
    assert jnp.allclose(X[:, 1], jnp.sin(U[:, 0]) * jnp.cos(U[:, 1]), atol=tol)
    assert jnp.allclose(X[:, 2], jnp.sin(U[:, 1]), atol=tol)
