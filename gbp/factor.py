import jax
import jax.numpy as jnp

@jax.jit
def h_fn(x):
    return x


@jax.jit
def h2_fn(xs):
    """
    xs: shape (2, D), where:
        - xs[0] is x1
        - xs[1] is x2
    """
    x1 = xs[0]
    x2 = xs[1]

    #jax.debug.print("Shape of x2 - x1: {}", (x2 - x1).shape)

    return x2 - x1


@jax.jit
def h3_fn(x):
    """
    Predicts measurement h(x) for coarse prior.

    Input:
        x: (8,) → 4 stacked fine-level variables (each 2D)
    Output:
        z_hat: (16,) = [x0, x1, x2, x3, x1-x0, x2-x0, x3-x1, x3-x2]
    """
    x = x.reshape(-1)
    
    x0 = x[0:2]
    x1 = x[2:4]
    x2 = x[4:6]
    x3 = x[6:8]

    # 4 priors (just xi)
    z_hat_0 = x0
    z_hat_1 = x1
    z_hat_2 = x2
    z_hat_3 = x3

    # 4 internal between
    z_hat_4 = x1 - x0
    z_hat_5 = x2 - x0
    z_hat_6 = x3 - x1
    z_hat_7 = x3 - x2

    return jnp.concatenate([
        z_hat_0, z_hat_1, z_hat_2, z_hat_3,
        z_hat_4, z_hat_5, z_hat_6, z_hat_7
    ])


@jax.jit
def h4_fn(xs):
    """
    Predicts coarse between measurement h(xs) where:
      - xs[0] is coarse variable i (8D)
      - xs[1] is coarse variable j (8D)
    Fixed: uses v01→v02 and v11→v12 edges
    
    Returns:
        z_hat: shape (4,) = two 2D relative positions
    """
    xi = xs[0]
    xj = xs[1]

    # First residual: xj[0:2] - xi[2:4]  (v02 - v01)
    r1 = xj[0:2] - xi[2:4]

    # Second residual: xj[4:6] - xi[6:8] (v12 - v11)
    r2 = xj[4:6] - xi[6:8]

    return jnp.concatenate([r1, r2])  # shape (4,)


@jax.jit
def h5_fn(xs):
    """
    Predicts coarse between measurement h(xs) where:
      - xs[0] is coarse variable i (8D)
      - xs[1] is coarse variable j (8D)
    Fixed: uses v10→v20 and v11→v21 edges

    Returns:
        z_hat: shape (4,) = two 2D relative positions
    """
    xi = xs[0]
    xj = xs[1]

    # First residual: xj[0:2] - xi[4:6]  (v20 - v10)
    r1 = xj[0:2] - xi[4:6]

    # Second residual: xj[2:4] - xi[6:8] (v21 - v11)
    r2 = xj[2:4] - xi[6:8]

    return jnp.concatenate([r1, r2])  # shape (4,)


@jax.jit
def h6_fn(x):
    """
    Predicts measurement h(x) for coarse prior.

    Input:
        x: (32,) → 4 stacked coarse-level varibales (each 8D)
    Output:
        z_hat: (80,) 
    """
    x = x.reshape(-1)
    
    x0 = x[0:2]
    x1 = x[2:4]
    x2 = x[4:6]
    x3 = x[6:8]
    x4 = x[8:10]
    x5 = x[10:12]
    x6 = x[12:14]               
    x7 = x[14:16]
    x8 = x[16:18]
    x9 = x[18:20]
    x10 = x[20:22]
    x11 = x[22:24]
    x12 = x[24:26]
    x13 = x[26:28]
    x14 = x[28:30]
    x15 = x[30:32]


    return jnp.concatenate([
        x0, x1, x2, x3,
        x1-x0, x2-x0, x3-x1, x3-x2,
        x4, x5, x6, x7,
        x5-x4, x6-x4, x7-x5, x7-x6,
        x8, x9, x10, x11,
        x9-x8, x10-x8, x11-x9, x11-x10,
        x12, x13, x14, x15,
        x13-x12, x14-x12, x15-x13, x15-x14,
        x4-x1, x6-x3,
        x8-x2, x9-x3,
        x12-x6,x13-x7,
        x12-x9,x14-x11])

@jax.jit
def h7_fn(xs):
    """
    Predicts coarser horizontal between measurement h(xs) where:
      - xs[0] is coarser variable i (32D)
      - xs[1] is coarser variable j (32D)
    Fixed: uses v01→v02 and v11→v12 edges
    
    Returns:
        z_hat: shape (8,) = four 2D relative positions
    """

    xi = xs[0]
    xj = xs[1]

    # First residual: 
    r1 = xj[0:2] - xi[10:12]

    # Second residual: 
    r2 = xj[4:6] - xi[14:16]

    # Third residual: 
    r3 = xj[16:18] - xi[26:28]

    # Forth residual
    r4 = xj[20:22] - xi[30:32]

    return jnp.concatenate([r1, r2, r3, r4])  # shape (8,)


@jax.jit
def h8_fn(xs):
    """
    Predicts coarse between measurement h(xs) where:
      - xs[0] is coarse variable i (32D)
      - xs[1] is coarse variable j (32D)
    Fixed: uses v10→v20 and v11→v21 edges

    Returns:
        z_hat: shape (8,) = four 2D relative positions
    """
    xi = xs[0]
    xj = xs[1]

    # First residual: 
    r1 = xj[0:2] - xi[20:22]

    # Second residual: 
    r2 = xj[2:4] - xi[22:24]

    # Third residual: 
    r3 = xj[8:10] - xi[28:30]

    # Forth residual
    r4 = xj[10:12] - xi[30:32]

    return jnp.concatenate([r1, r2, r3, r4])  # shape (8,)

