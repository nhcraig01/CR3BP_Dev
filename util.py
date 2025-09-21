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

CR3BP_dUdr_jax = jax.jit(jax.grad(CR3BP_U_jax, argnums=0))

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
        saveat=saveat
    )
    return sol.ys[-1]

CR3BP_dPhidX0_jax = jax.jacrev(CR3BP_Phi_jax, argnums=0)
    

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


def CR3BP_symOrb_solver(fv0: np.ndarray, mu: float, fixed_var: Literal["x0", "z0", "free"] = "free", ftol: float = 1e-12, rtol: float = 1e-13, atol: float = 1e-13, dt0 = None) -> tuple[np.ndarray, dict]:
    """ Solves for a symmetric periodic orbit in the Circular Restricted Three-Body Problem (CR3BP) using a root-finding approach.

    Parameters:
    fv0 (np.ndarray): Initial guess for the free variables defining the symmetric periodic orbit.
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x0', 'z0', or 'free').
    ftol(float): Tolerance for the root-finding algorithm.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    tuple: A tuple containing:
        - fv_sol (np.ndarray): Solution for the free variables defining the symmetric periodic orbit.
        - info (dict): Information about the root-finding process.
    """
    
    res = lambda fv: CR3BP_symOrb_f(fv,mu, fixed_var, rtol, atol, dt0)
    res_jac = lambda fv: CR3BP_symOrb_df(fv,mu, fixed_var, rtol, atol, dt0)

    fv_sol = least_squares(res, x0 = fv0, jac=res_jac, verbose=2, ftol=ftol,x_scale='jac')
    
    return fv_sol


