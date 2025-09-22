# util.py
# Utility functions for various operations within the orbit library generations

import yaml
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve, least_squares
import jax
import jax.numpy as jnp
import diffrax as dfx
from typing import Literal
from functools import partial
from numpy.linalg import norm
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment


"""
import warnings
warnings.filterwarnings("error", message=".*unhashable type: .*Tracer.*")

# optional: see full JAX frames too
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
"""
# JAX precision
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)


@partial(jax.jit)
def CR3BP_U_jax(r: jnp.ndarray, mu: float) -> float:
    """ Computes the pseudo-potential function U for the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    r (jnp.ndarray): Position vector in the rotating frame (x, y, z).
    mu (float): Mass ratio of the two primary bodies.

    Returns:
    float: The value of the pseudo-potential function U at position r.
    """
    x, y, z = r
    r1 = jnp.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = jnp.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    U = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
    return U

CR3BP_dUdr_jax = partial(jax.jit)(jax.grad(CR3BP_U_jax, argnums=0)) # 1st order Jacobians of U
CR3BP_dUdrdr_jax = partial(jax.jit)(jax.jacobian(CR3BP_dUdr_jax, argnums=0)) # 2nd order Jacobians of U

@partial(jax.jit)
def CR3BP_dfdX_jax(X: jnp.ndarray, mu: float) -> jnp.ndarray:
    """ Computes the Jacobian of the equations of motion for the Circular Restricted Three-Body Problem (CR3BP)

    Parameters:
    X (jnp.ndarray): State vector [x, y, z, vx, vy, vz].
    mu (float): Mass ratio of the two primary bodies.

    Returns:
    jnp.ndarray: Jacobian matrix of the equations of motion.
    """
    r = X[:3]
    v = X[3:]
    
    # Construct the Jacobian matrix

    dUdrdr = CR3BP_dUdrdr_jax(r, mu)  
    A = jnp.block([
        [jnp.zeros((3, 3)), jnp.eye(3)],
        [dUdrdr, jnp.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])]
    ])
    
    return A

@partial(jax.jit)
def CR3BP_EOMs_jax(X: jnp.ndarray, mu: float) -> jnp.ndarray:
    """ Computes the equations of motion for the Circular Restricted Three-Body Problem (CR3BP) in the rotating frame.

    Parameters:
    X (jnp.ndarray): State vector [x, y, z, vx, vy, vz].
    mu (float): Mass ratio of the two primary bodies.

    Returns:
    X_dot (jnp.ndarray): Time derivative of the state vector [vx, vy, vz, ax, ay, az].
    """
    r = X[:3]
    v = X[3:]
    
    dU = CR3BP_dUdr_jax(r, mu)
    
    ax = 2 * v[1] + dU[0]
    ay = -2 * v[0] + dU[1]
    az = dU[2]
    a = jnp.array([ax, ay, az])
    
    return jnp.concatenate((v, a))

@partial(jax.jit, static_argnames=["rtol","atol","dt0",])
def CR3BP_Phi_jax(X0: jnp.ndarray, T: float, mu: float, rtol: float = 1e-13, atol: float = 1e-13, dt0 = None) -> jnp.ndarray:
    """ CR3BP flow map using JAX and Diffrax.
    Integrates the equations of motion for the Circular Restricted Three-Body Problem (CR3BP) using JAX and Diffrax.

    Parameters:
    X0 (jnp.ndarray): Initial state vector [x, y, z, vx, vy, vz].
    T (float): Total integration time.
    mu (float): Mass ratio of the two primary bodies.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    Xf (jnp.ndarray): State vector at time T.
    """
    def ode(t, X, args):
        mu = args
        return CR3BP_EOMs_jax(X, mu)

    term = dfx.ODETerm(ode)
    solver = dfx.Dopri8()
    controller = dfx.PIDController(rtol=rtol, atol=atol)
    saveat = dfx.SaveAt(t1=True)
    
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T,
        dt0=dt0,
        y0=X0,
        args=mu,
        stepsize_controller=controller,
        max_steps=1_000_000,
        saveat=saveat
    )
    return sol.ys[-1]

CR3BP_dPhidX0_jax = partial(jax.jit, static_argnames=["rtol","atol","dt0",])(jax.jacrev(CR3BP_Phi_jax, argnums=0))
    
@partial(jax.jit, static_argnames=["rtol","atol","dt0",])
def CR3BP_dPhi_jax(X0: jnp.ndarray, T: float, mu: float, rtol: float = 1e-13, atol: float = 1e-13, dt0 = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ Computes the Jacobians of the flow map for the Circular Restricted Three-Body Problem (CR3BP) using JAX and Diffrax.

    Parameters:
    X0 (jnp.ndarray): Initial state vector [x, y, z, vx, vy, vz].
    T (float): Total integration time.
    mu (float): Mass ratio of the two primary bodies.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    tuple: A tuple containing:
        - dPhidX0 (jnp.ndarray): Jacobian of the flow map with respect to the initial state.
        - dPhidT (jnp.ndarray): Jacobian of the flow map with respect to time T.
    """
    
    Xf = CR3BP_Phi_jax(X0, T, mu, rtol, atol, dt0)
    dPhidX0 = CR3BP_dPhidX0_jax(X0, T, mu, rtol, atol, dt0)
    dPhidT = CR3BP_EOMs_jax(Xf, mu)
    return dPhidX0, dPhidT

def CR3BP_symOrb_f(fv: np.ndarray, mu: float, fixed_var: Literal["x0", "z0", "free"] = "free", rtol: float = 1e-13, atol: float = 1e-13, dt0 = None) -> np.ndarray:
    """ Computes the final state residual for a symmetric orbit solver in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    fv (np.ndarray): Free variables defining the symmetric periodic orbit.
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x0', 'z0', or 'free').
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    tuple: A tuple containing:
        - res (np.ndarray): residual of the final state at half the period.
        - jac (np.ndarray): Jacobian of the residual with respect to the free variables.
    """
    # Unpack free variables
    x0, z0, vy0, T2 = fv
    # Initial state vector
    X0 = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
    
    # Integrate to half the period
    Xf = CR3BP_Phi_jax(X0, T2, mu, rtol, atol, dt0)

    # Final state residual for symmetric orbit
    res = np.array([
        Xf[1],  # y should be 0
        Xf[3],  # vx should be 0
        Xf[5]   # vz should be 0
    ])    

    return res

def CR3BP_symOrb_df(fv: np.ndarray, mu: float, fixed_var: Literal["x0", "z0", "free"] = "free", rtol: float = 1e-13, atol: float = 1e-13, dt0 = None) -> np.ndarray:
    """ Computes the Jacobian of the final state residual for a symmetric orbit solver in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    fv (np.ndarray): Free variables defining the symmetric periodic orbit.
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x0', 'z0', or 'free').
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    jac (np.ndarray): Jacobian of the residual with respect to the free variables.
    """

    # Unpack free variables
    x0, z0, vy0, T2 = fv
    # Initial state vector
    X0 = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])

    # compute Jacobians of the flow map
    dPhidX0, dPhidT = CR3BP_dPhi_jax(X0, T2, mu, rtol, atol, dt0)
    
    #  Full Jacobian including time derivative
    jac_full = np.concatenate((dPhidX0, dPhidT[:, None]), axis=1)

    # Jacobian of the initial state with respect to free variables
    if fixed_var == "x0":
        dX0dFv = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=float)
    elif fixed_var == "z0":
        dX0dFv = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=float)
    elif fixed_var == "free":
        dX0dFv = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=float)
    
    # Jacobin of residual with respect to final state
    dres_dfv = np.array([[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=float)
    

    # Jacobian of the final state residual with respect to free variables
    jac = dres_dfv @ jac_full @ dX0dFv

    return jac

def CR3BP_symOrb_solver(fv0: np.ndarray, mu: float, fixed_var: Literal["x0", "z0", "free"] = "free", gtol: float = 1e-12, rtol: float = 1e-13, atol: float = 1e-13, dt0 = None) -> tuple[np.ndarray, dict]:
    """ Solves for a symmetric periodic orbit in the Circular Restricted Three-Body Problem (CR3BP) using a root-finding approach.

    Parameters:
    fv0 (np.ndarray): Initial guess for the free variables defining the symmetric periodic orbit.
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x0', 'z0', or 'free').
    gtol(float): Gradient tolerance for the root-finding algorithm.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    tuple: A tuple containing:
        - fv_sol (np.ndarray): Solution for the free variables defining the symmetric periodic orbit.
        - info (dict): Information about the root-finding process.
    """
    
    xtol = 1e-14
    ftol = 1e-14
    res = lambda fv: CR3BP_symOrb_f(fv,mu, fixed_var, rtol, atol, dt0)
    res_jac = lambda fv: CR3BP_symOrb_df(fv,mu, fixed_var, rtol, atol, dt0)

    fv_sol = least_squares(res, x0 = fv0, jac=res_jac, verbose=2,ftol=ftol,xtol=xtol,gtol=gtol,x_scale='jac')
    
    return fv_sol["x"]

def CR3BP_Lyap_ICs(Lagrn_pt: np.ndarray, mu: float, amp: float=1e-2) -> tuple[np.ndarray,float]:
    """ Computes ballpark initial conditions for a planar orbit around a Lagrange point via linearizing the EOMs.

    Parameters:
    Lagrn_pt (np.ndarray): Position of the Lagrange point [x, y, z].
    mu (float): Mass ratio of the two primary bodies.
    amp (float): Amplitude of the initial perturbation.

    Returns:
    np.ndarray: Initial state vector [x, y, z, vx, vy, vz] for the Lyapunov orbit.
    """
    # Compute the Jacobian of the EOMs at the Lagrange point
    X_eq = np.array([Lagrn_pt[0], Lagrn_pt[1], Lagrn_pt[2], 0.0, 0.0, 0.0])
    A = CR3BP_dfdX_jax(X_eq, mu)
    
    # Sort eigenvalues and eigenvectors
    w, V = eig(A)
    
    # Identify the planar Lyapunov eigenmode (purely imaginary eigenvalues)
    idx_lyap = np.flatnonzero((np.abs(w.imag) > 1e-6) & (np.abs(w.real) < 1e-6) & (np.abs(V[2,:]) < 1e-6))
    
    if len(idx_lyap) == 0:
        raise ValueError("No planar Lyapunov eigenmode found.")
    
    # Choose the first planar Lyapunov mode
    v_lyap = V[:, idx_lyap[0]].real
    
    # Set initial conditions based on the specified direction
    v = v_lyap/v_lyap[0]  # normalize so x-component is 1
    ICs = X_eq + v*amp  # perturb in x-direction
    T = 2 * np.pi / np.abs(w[idx_lyap[0]].imag)  # period of the linearized motion
    
    return np.array(ICs), T

def fix_phase(v: np.ndarray) -> np.ndarray:
    """ Fixes the phase of a vector v by normalizing it based on its first non-zero element.

    Parameters:
    v (np.ndarray): The vector whose phase needs to be fixed.

    Returns:
    np.ndarray: The phase-fixed vector.
    """
    k = np.flatnonzero(np.abs(v) > 0)
    if k.size:
        a = v[k[0]]
        v = v * (a.conjugate() / abs(a))
    return v

def fix_phase_all(V: np.ndarray) -> np.ndarray:
    """Apply fix_phase to every eigenvector (column) independently.

    Parameters:
    V (np.ndarray): Matrix whose columns are eigenvectors.

    Returns:
    np.ndarray: Matrix with phase-fixed eigenvectors as columns.
    """

    W = V.copy()
    for j in range(W.shape[1]):
        W[:, j] = fix_phase(W[:, j])
    return W

def pair_reciprocals(w: np.ndarray, v: np.ndarray, tol_recip: float=1e-8, tol_unit: float=1e-8) -> tuple[np.ndarray, np.ndarray]:
    """ Pairs eigenvalues and eigenvectors that are reciprocals of each other
    
    Parameters:
    w (np.ndarray): Array of eigenvalues.
    v (np.ndarray): Matrix whose columns are eigenvectors.
    tol_recip (float): Tolerance for determining if two eigenvalues are reciprocals.
    tol_unit (float): Tolerance for determining if an eigenvalue is on the unit circle.

    Returns:
    tuple: A tuple containing:
        - w_paired (np.ndarray): Eigenvalues paired as reciprocals.
        - v_paired (np.ndarray): Corresponding eigenvectors.
    """
    n = len(w)
    unused = np.ones(n,dtype=bool)
    order = list[int] ()

    def recip_dst(lam1, lam2):
        return abs(lam1 * lam2 - 1.0)
    
    # Group reciprocals together
    for i in range(int(n/2)):
        order.append(np.flatnonzero(unused)[0])  # first unused
        unused[order[-1]] = 0
        lam = w[order[-1]]
        # find its reciprocal partner
        partner = np.flatnonzero(unused & (recip_dst(w, lam) < tol_recip))
        order.append(partner[0])
        unused[partner[0]] = 0

    # Order reciprocals within pair
    for i in range(0, n, 2):
        lam1 = w[order[i]]
        if abs(lam1) > 1: # vals non on unit circle and |λ|>1 first
            continue
        elif abs(lam1) < 1: # vals non on unit circle and |λ|<1 second
            # reverse indices of pair
            order[i], order[i+1] = order[i+1], order[i]
        elif abs(abs(lam1)-1) < tol_unit: # vals on unit circle, order by angle
            if np.angle(lam1)<0:
                # reverse indices of pair
                order[i], order[i+1] = order[i+1], order[i]

    w_paired = w[order]
    v_paired = v[:, order]
    return w_paired, v_paired

def eig_sort(A: np.ndarray,prev_vecs: np.ndarray | None=None, tol_one: float=1e-4, tol_unit: float=1e-4, tol_recip: float=1e-8) -> tuple[np.ndarray, np.ndarray]:
    """ Sorts the eigenvalues and eigenvectors of a matrix A to maintain continuity with previous eigenvectors. 
    Following rules are established if no previous basis is given:
    1. Vals and Vecs paired as reciprocals
    2. Order by |λ|≈1 first, then angle off real.
    3. Order by real vals next, then complex.
    If previous basis is given, the new basis is chosen to maximize overlap with previous basis.
    
    Parameters:
    A (np.ndarray): The input matrix.
    prev_vecs (np.ndarray): The previous eigenvectors for continuity.

    Returns:
    tuple: A tuple containing:
        - w_out (np.ndarray): Sorted eigenvalues.
        - V_out (np.ndarray): Sorted eigenvectors.
    """
    w, V = eig(A) # w: eigenvalues, V: eigenvectors as columns
    n = len(w)
    V = fix_phase_all(V) # fix phases for readability
        

    if prev_vecs is None: # no previous basis: do a stable, physics-aware sort
        # Pair reciprocals first
        w, V = pair_reciprocals(w, V, tol_recip=tol_recip,tol_unit=tol_unit)

        # Order reciprocal paris
        lead = w[::2]  # leading eigenvalue of each pair
        idx_one = 2*np.flatnonzero((np.abs(lead - 1.0) < tol_one) | (np.abs(lead + 1.0) < tol_one)) # indices of pairs with λ≈1
        idx_unit = 2 * np.flatnonzero((np.abs(np.abs(lead) - 1.0) < tol_unit) & ~(np.isin(2*np.arange(len(lead)), idx_one))) # indices of pairs with |λ|≈1 but not λ≈1
        idx_real = 2 * np.flatnonzero((np.abs(lead.imag) < tol_unit) & ~(np.isin(2*np.arange(len(lead)), idx_one))) # indices of pairs with real λ but not λ≈1
        idx_other = 2 * np.flatnonzero(~np.isin(2*np.arange(len(lead)), np.concatenate([idx_one, idx_unit, idx_real]))) # indices of all other pairs

        order = np.concatenate([idx_one, idx_unit, idx_real, idx_other])
        # print("Indices of pairs with λ≈±1:", idx_one)
        # print("Indices of pairs with |λ|≈1:", idx_unit)
        # print("Indices of pairs with real λ:", idx_real)
        # print("Indices of other pairs:", idx_other)
        # print("Order of pairs:", order)
        order = np.column_stack([order, order+1]).flatten()  # include both of each pair


        w_out = w[order]
        V_out = V[:, order]
    else:
        # With previous basis: maximize overlap |<v_prev_i, v_new_j>|
        prev = prev_vecs
        # normalize columns
        P = prev / np.maximum(norm(prev, axis=0, keepdims=True), 1e-30)
        Q = V    / np.maximum(norm(V,    axis=0, keepdims=True), 1e-30)
        S = np.abs(P.conj().T @ Q)  # similarity matrix (n x n)

        # Hungarian assignment to maximize total overlap (use -S to minimize)
        row, col = linear_sum_assignment(-S)

        order_by_prev = np.argsort(row)
        col_in_prev_order = col[order_by_prev]

        # reorder and phase-align
        w_out = w[col_in_prev_order]
        V_out = V[:, col_in_prev_order].copy()

    return w_out, V_out