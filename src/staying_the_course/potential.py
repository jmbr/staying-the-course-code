import jax
import jax.numpy as jnp


@jax.jit
def potential2d(z):
    """Original Muller potential."""
    return -200.0*jnp.exp(-(z[0]-1.0)**2-10.0*z[1]**2)-100.0*jnp.exp(-z[0]**2-10.0*(z[1]-1/2)**2)-170.0*jnp.exp(-(13/2)*(z[0]+1/2)**2+11.0*(z[0]+1/2)*(z[1]-3/2)-(13/2)*(z[1]-3/2)**2)+15.0*jnp.exp((7/10)*(z[0]+1)**2+(3/5)*(z[0]+1.0)*(z[1]-1.0)+(7/10)*(z[1]-1.0)**2)  # noqa


@jax.jit
def polar_to_cartesian(u):
    return jnp.array([1.973521294 * u[0] - 1.85,
                      1.750704373 * u[1] + 0.875])


@jax.jit
def cartesian_to_polar(z):
    return jnp.array([0.5067084926 * z[0] + 0.9374107113,
                      0.5711986646 * z[1] - 0.4997988314])


@jax.jit
def phi1(x):
    """Stereographic projection (from North pole)."""
    return x[:2] / (1.0 - x[2])


@jax.jit
def phi1_inv(p):
    """Inverse of stereographic projection (from North pole)."""
    p2 = jnp.dot(p, p)
    return jnp.array([2.0 * p[0], 2.0 * p[1], p2 - 1.0]) / (p2 + 1.0)


@jax.jit
def phi2_inv(q):
    """Inverse of stereographic projection (from South pole)."""
    q2 = jnp.dot(q, q)
    return jnp.array([2.0 * q[0], 2.0 * q[1], 1.0 - q2]) / (q2 + 1.0)


@jax.jit
def potential3d(x):
    """Muller potential in 3D coordinates."""
    u = jnp.array([jnp.arctan2(x[1], x[0]),
                   jnp.arctan2(x[2], jnp.sqrt(x[0]**2 + x[1]**2))])
    return potential2d(polar_to_cartesian(u))


@jax.jit
def potential_stereographic1(p):
    """Muller potential on stereographic projection (North pole)."""
    return potential3d(phi1_inv(p))


@jax.jit
def potential_stereographic2(p):
    """Muller potential on stereographic projection (South pole)."""
    return potential3d(phi2_inv(p))


@jax.jit
def force_stereographic1(p):
    return -jax.jacobian(potential_stereographic1)(p)


# Fixed points of the Müller potential in Cartesian 2D coordinates.
muller_fixed_points = jnp.array(
    [[0.623499404930877, 0.0280377585286857],
     [0.212486582000662, 0.292988325107368],
     [-0.0500108229982061, 0.466694104871972],
     [-0.822001558732732, 0.624312802814871],
     [-0.558223634633024, 1.44172584180467]])

# Fixed points of the Müller potential in Cartesian 3D coordinates on the
# sphere.
muller_fixed_points3d = jnp.array(
    [[0.27632632, 0.84100892, -0.46513199],
     [0.47435655, 0.81760542, -0.32635446],
     [0.59553828, 0.76936331, -0.23111522],
     [0.85849672, 0.49256384, -0.14270336],
     [0.75211518, 0.5771847, 0.31808894]])
