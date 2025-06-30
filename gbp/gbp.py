import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from functools import partial
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

@register_pytree_node_class
class Gaussian:
    def __init__(self, eta, Lam):
        self.eta = eta
        self.Lam = Lam

    def tree_flatten(self):
        return (self.eta, self.Lam), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    
    def mu(self):
        return jnp.where(
            jnp.allclose(self.Lam, 0), self.eta, jnp.linalg.solve(self.Lam, self.eta)
        )
    
    def sigma(self):
        return jnp.linalg.inv(self.Lam)

    def zero_like(self):
        return Gaussian(jnp.zeros_like(self.eta), jnp.zeros_like(self.Lam))

    def __repr__(self) -> str:
        return f"Gaussian(eta={self.eta}, lam={self.Lam})"

    def __mul__(self, other):
        return Gaussian(self.eta + other.eta, self.Lam + other.Lam)

    def __truediv__(self, other):
        return Gaussian(self.eta - other.eta, self.Lam - other.Lam)

    def copy(self):
        return Gaussian(self.eta.copy(), self.Lam.copy())


@register_pytree_node_class
class Variable:
    var_id: int
    belief: Gaussian
    msgs: Gaussian
    adj_factor_idx: jnp.array

    def __init__(self, var_id, belief, msgs, adj_factor_idx):
        self.var_id = var_id
        self.belief = belief
        self.msgs = msgs
        self.adj_factor_idx = adj_factor_idx

    def tree_flatten(self):
        return (self.var_id, self.belief, self.msgs, self.adj_factor_idx), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class Factor:
    factor_id: jnp.array
    z: jnp.ndarray
    z_Lam: jnp.ndarray
    threshold: jnp.ndarray
    potential: Gaussian
    adj_var_id: jnp.array
    adj_var_idx: jnp.array

    def __init__(
        self, factor_id, z, z_Lam, threshold, potential, adj_var_id, adj_var_idx
    ):
        self.factor_id = factor_id
        self.z = z
        self.z_Lam = z_Lam
        self.threshold = threshold
        self.potential = potential
        self.adj_var_id = adj_var_id
        self.adj_var_idx = adj_var_idx

    def tree_flatten(self):
        return (
            self.factor_id,
            self.z,
            self.z_Lam,
            self.threshold,
            self.potential,
            self.adj_var_id,
            self.adj_var_idx,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@partial(jax.jit, static_argnames=["i", "j"])
def marginalize(gaussians: Gaussian, i, j): # Equ. (46), (47); Compute msg to i:j Variables from connected factors
    eta = gaussians.eta
    Lam = gaussians.Lam
    k = eta.size
    idx = jnp.arange(0, k)
    aa = idx[i:j] # index from i to j-1
    bb = jnp.concatenate([idx[:i], idx[j:]]) # rest
    aa_eta = eta[aa]
    bb_eta = eta[bb]
    aa_Lam = Lam[aa[:, None], aa]
    ab_Lam = Lam[aa[:, None], bb]
    bb_Lam = Lam[bb][:, bb]
    if bb_Lam.size == 0:
        return Gaussian(aa_eta, aa_Lam)
    # print("How large? ", bb_Lam.shape)

    bb_Cov = jnp.linalg.inv(bb_Lam)
    eta = aa_eta - ab_Lam @ bb_Cov @ bb_eta
    Lam = aa_Lam - ab_Lam @ bb_Cov @ ab_Lam.T
    return Gaussian(eta, Lam)




@partial(jax.jit, static_argnames=["axis"])
def tree_stack(tree, axis=0):
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v, axis=axis), *tree)




@jax.jit
def update_belief(var: Variable, ftov_msgs): # Calculate Eq. (7)
    belief = var.belief.zero_like()
    for i in range(ftov_msgs.eta.shape[0]):
        belief = belief * Gaussian(ftov_msgs.eta[i], ftov_msgs.Lam[i])


    return belief


@jax.jit
def compute_vtof_msgs(var: Variable, ftov_msgs): # Eq.(19); do for each variable (x_m)
    vtof_msgs = []
    for i, idx in enumerate(var.adj_factor_idx): # for each f_si connected to x_m...
        msg = var.belief / Gaussian(ftov_msgs.eta[i], ftov_msgs.Lam[i]) # Eq.(19) LHS subscript of SUM
        eta = jnp.where(idx < 0, msg.zero_like().eta, msg.eta) # Those not connected should not affect the calculation (idx < 0)
        Lam = jnp.where(idx < 0, msg.zero_like().Lam, msg.Lam) # The reason to not using "if" (while it's per-element) is to optimize better
        vtof_msgs.append(Gaussian(eta, Lam)) # append (x_m -> f_si)
    return tree_stack(vtof_msgs) # [(x_m -> f_s1), (x_m -> f_s2), ... ] # The length is Ni_v



@partial(jax.jit, static_argnames=["h_fn"])
def factor_energy(factor, xs, h_fn):
    h = h_fn(xs)
    z = factor.z
    z_Lam = factor.z_Lam
    r = z - h
    return 0.5 * r @ z_Lam @ r.T



@partial(jax.jit, static_argnames=["h_fn", "w"])
def factor_update(factor, xs, h_fn, w):
    h = h_fn(xs)
    J = jax.jacrev(h_fn)(xs).reshape(h.size, xs.size) # Jacobian auto-diff (J_s)

    z = factor.z # I think this is a vector
    z_Lam = factor.z_Lam
    
    r = z - h.reshape(-1) # TODO: reshape can be problematic
    s = w(r.T @ z_Lam @ r, factor.threshold) # Scale to consider Robust Loss
    Lam = s * J.T @ z_Lam @ J # Eq. (36)
    eta = s * J.T @ z_Lam @ (J @ xs.reshape(-1) + r) # TODO: reshape can be problematic; Eq. (36); xs should be a vector
    return Gaussian(eta, Lam) # Factor; represented w.r.t. neighboring variables xs



@jax.jit
def compute_ftov_msg(factor, vtof_msgs): # Ch 3.5 Message Passing at a Factor Node
    # vtof_msgs: Variable to Factor Messages; shape: (N_adj, dim) 
    # where N_adj is the number of adjacent variables and dim is the dimension of each variable

    N_adj, dim = vtof_msgs.eta.shape

    
    pot = factor.potential.copy() # log(f_s), but for only a specific variable a factor is connected to.
    i = 0
    for n in range(N_adj): # Add all! (Produce all)
        j = i + dim
        pot.eta = pot.eta.at[i:j].add(vtof_msgs.eta[n])
        pot.Lam = pot.Lam.at[i:j, i:j].add(vtof_msgs.Lam[n])
        i = j


    ftov_msgs = []
    i = 0
    for n in range(N_adj):
        j = i + dim
        pot_m_1 = pot.copy()
        pot_m_1.eta = pot_m_1.eta.at[i:j].add(-vtof_msgs.eta[n]) # Subtract direction of going out! (42)
        pot_m_1.Lam = pot_m_1.Lam.at[i:j, i:j].add(-vtof_msgs.Lam[n]) # (43)
        msg = marginalize(pot_m_1, i, j) # (46), (47)
        ftov_msgs.append(msg)
        i = j

    
    return tree_stack(ftov_msgs)


@jax.jit
def update_variable(varis): # Update belief with receiving msgs and calculate msg to factors; varis.msgs are up-to-date and varis.belief are not
    varis.belief = jax.vmap(update_belief)(varis, varis.msgs) # Eq. (7); varis.msgs is receiving msgs (ftov)
    vtof_msgs = jax.vmap(compute_vtof_msgs)(varis, varis.msgs) # Variable -> Factor Msg; Eq. (19)
    linpoints = jax.vmap(lambda x: x.mu())(varis.belief) # Current avg of belief! Belief is posterior

    return varis, vtof_msgs, linpoints # vtof msgs: # Var * # Var-direction (factor, Ni_v) msgs


@partial(jax.jit, static_argnames=["f", "w"])
def update_factor(facs, varis, vtof_msgs, linpoints, f, w): # f is factor function, w is robustifier
    vtof_msgs_reordered = jax.tree_util.tree_map( # Variable to factor messages to specific (variable, factor; or variable-direction) pair
        lambda x: x[facs.adj_var_id, facs.adj_var_idx], vtof_msgs # id: Variable id (one end), idx: direction (another end)
    )
    linpoints_reordered = jax.tree_util.tree_map(
        lambda x: x[facs.adj_var_id], linpoints # Reorder linpoints by adj_var_id: variables' mean for factors' one ends
    )
    
    facs.potential = jax.vmap(factor_update, in_axes=(0, 0, None, None))( # Calculate each factor potential (f_s(x, x_1, ..., x_M) of Eq. (15))
        facs, linpoints_reordered, f, w # Each factor contribution of variable-direction pair (factor: variable-direction pair)
    ) # 1 or 2-dimensional!! (gradient / prior factor or smoothness factor)
    ftov_msgs = jax.vmap(compute_ftov_msg)(facs, vtof_msgs_reordered) # ftov calculation by Eq. (15), with potential f_s, and msg vtof

    varis.msgs.eta = varis.msgs.eta.at[facs.adj_var_id, facs.adj_var_idx].set( # Setting varis' receiving messages
        ftov_msgs.eta
    )
    varis.msgs.Lam = varis.msgs.Lam.at[facs.adj_var_id, facs.adj_var_idx].set(
        ftov_msgs.Lam
    )

    mask = (varis.adj_factor_idx >= 0)[..., None]
    varis.msgs.eta = varis.msgs.eta * mask
    varis.msgs.Lam = varis.msgs.Lam * mask[..., None]

    return facs, varis


@jax.jit
def huber(e, t):
    x = jnp.sqrt(e)
    return jnp.where(x <= t, 1.0, t / x)


@jax.jit
def l2(e, _):
    return 1.0

