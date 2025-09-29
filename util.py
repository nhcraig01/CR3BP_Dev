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
from numpy.linalg import norm, det
from scipy.linalg import eig
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
from numpy.linalg import svd
import sympy as sp
import h5py


"""
import warnings
warnings.filterwarnings("error", message=".*unhashable type: .*Tracer.*")

# optional: see full JAX frames too
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
"""
# JAX precision
jax.config.update("jax_enable_x64", True)

def CR3BP_EOMs_Eval():
    x, y, z, vx, vy, vz = sp.symbols('x y z vx vy vz', real=True)
    mu = sp.symbols('mu', real=True, positive=True)

    r1 = sp.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = sp.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    U = 0.5*(x**2 + y**2) + (1 - mu) / r1 + mu / r2

    r = sp.Matrix([[x],
                   [y], 
                   [z]])
    v = sp.Matrix([[vx], 
                   [vy], 
                   [vz]])
    X = sp.Matrix([[r],
                   [v]])

    dUdr = sp.Matrix([U]).jacobian(r)

    #ax = 2*vy+dUdr[0]
    #ay = -2*vx+dUdr[1]
    #az = dUdr[2]

    ax = x+2*vy-((1-mu)*(x+mu))/(r1**3) - (mu*(x-1+mu))/(r2**3)
    ay = y-2*vx-((1-mu)*y)/(r1**3) - (mu*y)/(r2**3)
    az = -((1-mu)*z)/(r1**3) - (mu*z)/(r2**3)
    a = sp.Matrix([[ax], 
                   [ay], 
                   [az]])

    X_dot = sp.Matrix([[v],
                     [a]])
    
    CR3BP_EOMs_sym = sp.lambdify([X,mu], X_dot, 'jax')
    CR3BP_dfdX_sym = sp.lambdify([X,mu], X_dot.jacobian(X), 'jax')
    CR3BP_U_sym = sp.lambdify([r,mu],U, 'jax')
    CR3BP_dUdr_sym = sp.lambdify([r,mu],dUdr, 'jax')

    CR3BP_EOMs  = lambda X, mu: jnp.squeeze(jnp.asarray(CR3BP_EOMs_sym(X, mu)), -1)
    CR3BP_dfdX  = lambda X, mu: jnp.asarray(CR3BP_dfdX_sym(X, mu))  
    CR3BP_U     = lambda r, mu: jnp.asarray(CR3BP_U_sym(r, mu))
    CR3BP_dUdr  = lambda r, mu: jnp.squeeze(jnp.asarray(CR3BP_dUdr_sym(r, mu)), -1)


    return CR3BP_EOMs, CR3BP_dfdX, CR3BP_U, CR3BP_dUdr

CR3BP_EOMs, CR3BP_dfdX, CR3BP_U, CR3BP_dUdr = CR3BP_EOMs_Eval()

@partial(jax.jit)
def CR3BP_U_jax(r: jnp.array, mu: float) -> float:
    """ Computes the pseudo-potential function U for the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    r (jnp.array): Position vector in the rotating frame (x, y, z).
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

@partial(jax.jit, static_argnames=["rtol","atol",])
def CR3BP_Phi_jax(X0: jnp.array, T: float, mu: float, rtol: float = 1e-13, atol: float = 1e-13, t_hst_out: jnp.array = None) -> jnp.array:
    """ CR3BP flow map using JAX and Diffrax.
    Integrates the equations of motion for the Circular Restricted Three-Body Problem (CR3BP) using JAX and Diffrax.

    Parameters:
    X0 (jnp.array): Initial state vector [x, y, z, vx, vy, vz].
    T (float): Total integration time.
    mu (float): Mass ratio of the two primary bodies.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.
    t_hst_out(jnp.array): Array of output times if provided

    Returns:
    Xf (jnp.array): State vector at time T.
    """
    def ode(t, X, args):
        mu = args
        return CR3BP_EOMs(X, mu)

    term = dfx.ODETerm(ode)
    solver = dfx.Dopri8()
    controller = dfx.PIDController(rtol=rtol, atol=atol)

    if t_hst_out is None:
        saveat = dfx.SaveAt(t1=True)
    else:
        saveat = dfx.SaveAt(ts = t_hst_out)
    
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T,
        dt0=T/1e4,
        y0=X0,
        args=mu,
        stepsize_controller=controller,
        max_steps=5_000_000,
        saveat=saveat
    )
    if t_hst_out is None:
        return sol.ys[-1]
    else:
        return sol.ys

CR3BP_dPhidX0_jax = partial(jax.jit, static_argnames=["rtol","atol",])(jax.jacrev(CR3BP_Phi_jax, argnums=0))
    
def CR3BP_Traj_Sol(X0: np.array, T: float, mu: float, rtol: float = 1e-13, atol: float = 1e-13, Npts: int = 2000) -> tuple[np.array, np.array]:
    """ CR3BP Trajectory evaluation
    Integrates the equations of motion for the Circular Restricted Three-Body Problem (CR3BP) for a trajectory

    Parameters:
    X0 (np.array): Initial state vector [x, y, z, vx, vy, vz].
    T (float): Total integration time.
    mu (float): Mass ratio of the two primary bodies.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.
    Npts (int): Number of points to use in the time grid.

    Returns:
    X_hst (np.array): Array of state vectors at each time step.
    t_hst (np.array): Array of time steps.
    """

    def ode(t, X):
        return np.asarray(CR3BP_EOMs(X, mu)).reshape(-1)

    t_hst = np.linspace(0.0, T, Npts)
    sol = solve_ivp(
        fun = ode, 
        t_span = [0.0,T], 
        y0 = X0, 
        method = "DOP853", 
        t_eval = t_hst, 
        rtol = rtol, 
        atol = atol, 
        max_step=1e-3, 
        first_step = 1e-5)
    
    X_hst = sol.y
    return X_hst.T, t_hst

@partial(jax.jit, static_argnames=["rtol","atol",])
def CR3BP_dPhi_jax(X0: jnp.array, T: float, mu: float, rtol: float, atol: float) -> tuple[jnp.array, jnp.array]:
    """ Computes the Jacobians of the flow map for the Circular Restricted Three-Body Problem (CR3BP) using JAX and Diffrax.

    Parameters:
    X0 (jnp.array): Initial state vector [x, y, z, vx, vy, vz].
    T (float): Total integration time.
    mu (float): Mass ratio of the two primary bodies.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    tuple: A tuple containing:
        - dPhidX0 (jnp.array): Jacobian of the flow map with respect to the initial state.
        - dPhidT (jnp.array): Jacobian of the flow map with respect to time T.
    """
    
    Xf = CR3BP_Phi_jax(X0, T, mu, rtol, atol)
    dPhidX0 = CR3BP_dPhidX0_jax(X0, T, mu, rtol, atol)
    dPhidT = CR3BP_EOMs(Xf, mu)
    return dPhidX0, dPhidT

def CR3BP_symOrb_f(fv: np.array, mu: float, rtol: float = 1e-13, atol: float = 1e-13) -> np.array:
    """ Computes the final state residual for a symmetric orbit solver in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    fv (np.array): Free variables defining the symmetric periodic orbit.
    mu (float): Mass ratio of the two primary bodies.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    res (np.array): residual of the final state at half the period.
    """
    # Unpack free variables
    x0, z0, vy0, T2 = fv
    # Initial state vector
    X0 = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])
    
    # Integrate to half the period
    Xf = CR3BP_Phi_jax(X0, T2, mu, rtol, atol)

    # Final state residual for symmetric orbit
    res = np.array([
        Xf[1],  # y should be 0
        Xf[3],  # vx should be 0
        Xf[5]   # vz should be 0
    ])    

    return res

def CR3BP_symOrb_df(fv: np.array, mu: float, fixed_var: Literal["x", "z", "free"] = "free", rtol: float = 1e-13, atol: float = 1e-13) -> np.array:
    """ Computes the Jacobian of the final state residual for a symmetric orbit solver in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    fv (np.array): Free variables defining the symmetric periodic orbit.
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x0', 'z0', or 'free').
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    jac (np.array): Jacobian of the residual with respect to the free variables.
    """

    # Unpack free variables
    x0, z0, vy0, T2 = fv
    # Initial state vector
    X0 = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])

    # compute Jacobians of the flow map
    dPhidX0, dPhidT = CR3BP_dPhi_jax(X0, T2, mu, rtol, atol)
    
    #  Full Jacobian including time derivative
    dXfdX0T = np.concatenate((dPhidX0, dPhidT[:, None]), axis=1)

    # Jacobian of the initial state with respect to free variables
    if fixed_var == "x":
        dX0Tdfv = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=float)
    elif fixed_var == "z":
        dX0Tdfv = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=float)
    elif fixed_var == "free":
        dX0Tdfv = np.array([[1, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]], dtype=float)
    
    # Jacobin of residual with respect to final state
    dfdXf = np.array([[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=float)
    

    # Jacobian of the final state residual with respect to free variables
    dfdfv = dfdXf @ dXfdX0T @ dX0Tdfv

    return dfdfv

def CR3BP_symOrb_solver(fv0: np.array, mu: float, fixed_var: Literal["x", "z", "free"] = "free", gtol: float = 1e-12, rtol: float = 1e-13, atol: float = 1e-13):
    """ Solves for a symmetric periodic orbit in the Circular Restricted Three-Body Problem (CR3BP) using a root-finding approach.

    Parameters:
    fv0 (np.array): Initial guess for the free variables defining the symmetric periodic orbit.
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x0', 'z0', or 'free').
    gtol(float): Gradient tolerance for the root-finding algorithm.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.
    dt0 (float): Intial integration step size

    Returns:
    tuple: A tuple containing:
        - fv_sol (np.array): Solution for the free variables defining the symmetric periodic orbit.
        - info (dict): Information about the root-finding process.
    """
    
    xtol = 1e-12
    ftol = 1e-12
    res = lambda fv: CR3BP_symOrb_f(fv,mu, rtol, atol)
    res_jac = lambda fv: CR3BP_symOrb_df(fv,mu, fixed_var, rtol, atol)

    fv_sol = least_squares(res, 
                           x0 = fv0, 
                           jac=res_jac, 
                           verbose=0, 
                           ftol=ftol, 
                           xtol=xtol, 
                           gtol=gtol, 
                           x_scale='jac', 
                           method='dogbox',
                           max_nfev=1e4)
    
    return fv_sol

def CR3BP_Zplane_Cross_ICs(X0: np.array, T: float, mu, z_cross: float = 0.1, dir: int = 0, rtol: float = 1e-13, atol: float = 1e-13):
    """ Given the ICs of a periodic orbit under CR3BP dynamics, this function solves for the X/Z plane crossings at z=z_cross

    Parameters: 
    X0 (np.array): Periodic orbit initial state
    T (float): Periodic orbit period
    z_cross (float): Desired X/Y plane crossing z value
    dir (int): direction of crossing 0 (both), +1 (upward), -1 (downward)
    rtol (float): relative integration tolerance
    atol (float): absolute integration tolerance

    Returns:


    """
    def ode(t,X): return CR3BP_EOMs(X,mu)

    # event
    def ev_z(t,X):
        return X[2] - z_cross
    ev_z.terminal = False
    ev_z.direction = dir

    sol = solve_ivp(
        fun = ode, 
        t_span = [0.0,T], 
        y0 = X0, 
        method = "DOP853",
        rtol = rtol, 
        atol = atol, 
        max_step=T/100, 
        first_step = 1e-4, 
        events = ev_z)
    
    X0_cross = np.array(sol.y_events[0]).reshape(-1)

    return X0_cross

def CR3BP_asymOrb_f(fv: np.array, z0: float, mu: float, rtol: float = 1e-13, atol: float = 1e-13) -> np.array:
    """ Computes the final state residual for an asymmetric orbit solver in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    fv (np.array): Free variables defining the asymmetric orbit
    z0 (float): Orbit initial state z coordinate
    mu (float): Mass ratio of the two primary bodies.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    res (np.array): residual of the final state at the final period
    """
    # Unpack free variables
    x0, y0, vx0, vy0, vz0, T = fv
    X0 = np.array([x0,y0,z0,vx0,vy0,vz0]) # Intial state
    
    # Integrate to the period
    Xf = CR3BP_Phi_jax(X0, T, mu, rtol, atol)

    # Final state residual for asymmetric orbit
    res = Xf-X0  

    return res

def CR3BP_asymOrb_df(fv: np.array, z0: float, mu: float, fixed_var: Literal["x", "y", "free"] = "free", rtol: float = 1e-13, atol: float = 1e-13) -> np.array:
    """ Computes the Jacobian of the final state residual for an asymmetric orbit solver in the Circular Restricted Three-Body Problem (CR3BP).
    This already assumes that z0 is fixed at a constant value

    Parameters:
    fv (np.array): Free variables defining the asymmetric periodic orbit.
    z0 (float): Orbit initial state z coordinate
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x', 'y', or 'free').
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.

    Returns:
    jac (np.array): Jacobian of the residual with respect to the free variables.
    """

    # Unpack free variables
    x0, y0, vx0, vy0, vz0, T = fv
    X0 = np.array([x0,y0,z0,vx0,vy0,vz0]) # Intial state

    # compute Jacobians of the flow map
    dPhidX0, dPhidT = CR3BP_dPhi_jax(X0, T, mu, rtol, atol)
    
    #  Full Jacobian including time derivative
    dXfdX0T = np.concatenate((dPhidX0-np.identity(6), dPhidT[:, None]), axis=1)

    # Jacobian of the initial state with respect to free variables
    if fixed_var == "x":
        dX0Tdfv = np.array([[0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=float)
    elif fixed_var == "y":
        dX0Tdfv = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=float)
    elif fixed_var == "free":
        dX0Tdfv = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]], dtype=float)
    

    # Jacobian of the final state residual with respect to free variables
    jac = dXfdX0T @ dX0Tdfv

    return jac

def CR3BP_asymOrb_solver(fv0: np.array, z0: float, mu: float, fixed_var: Literal["x", "y", "free"] = "free", gtol: float = 1e-12, rtol: float = 1e-13, atol: float = 1e-13):
    """ Solves for an asymmetric periodic orbit in the Circular Restricted Three-Body Problem (CR3BP) using a root-finding approach.

    Parameters:
    fv0 (np.array): Initial guess for the free variables defining the symmetric periodic orbit.
    z0 (float): Orbit initial state z coordinate
    mu (float): Mass ratio of the two primary bodies.
    fixed_var (str): Specifies which variable is fixed ('x', 'y', or 'free').
    gtol(float): Gradient tolerance for the root-finding algorithm.
    rtol (float): Relative tolerance for the integrator.
    atol (float): Absolute tolerance for the integrator.
    dt0 (float): Intial integration step size

    Returns:
    tuple: A tuple containing:
        - fv_sol (np.array): Solution for the free variables defining the symmetric periodic orbit.
        - info (dict): Information about the root-finding process.
    """
    
    xtol = 3e-16 
    ftol = 1e-12
    f = lambda fv: CR3BP_asymOrb_f(fv,z0,mu,rtol,atol)
    f_jac = lambda fv: CR3BP_asymOrb_df(fv,z0,mu,fixed_var,rtol,atol)

    fv_sol = least_squares(f, 
                           x0 = fv0, 
                           jac=f_jac, 
                           verbose=0, 
                           ftol=ftol, 
                           xtol=xtol, 
                           gtol=gtol, 
                           x_scale='jac', 
                           method='lm',
                           max_nfev=int(1e4))
    
    return fv_sol

def CR3BP_Orb_solver_funcs(Solver_Type: Literal["Symmetric","Asymmetric"],z0 = None):
    if Solver_Type == "Symmetric":
        def X2fv(X, T): return np.array([X[0], X[2], X[4], T/2])
        def fv2X_T(fv): return np.array([fv[0], 0, fv[1], 0, fv[2], 0]), fv[3]*2

        CR3BP_Orb_solver = CR3BP_symOrb_solver
        CR3BP_f = CR3BP_symOrb_f
        CR3BP_df = CR3BP_symOrb_df
    elif Solver_Type == "Asymmetric":
        def X2fv(X,T): return np.array([X[0],X[1],X[3],X[4],X[5],T])
        def fv2X_T(fv): return np.array([fv[0],fv[1],z0,fv[2],fv[3],fv[4]]), float(fv[5])

        CR3BP_Orb_solver = partial(CR3BP_asymOrb_solver,z0=z0)
        CR3BP_f = partial(CR3BP_asymOrb_f,z0=z0)
        CR3BP_df = partial(CR3BP_asymOrb_df,z0=z0)
    
    return X2fv, fv2X_T, CR3BP_Orb_solver, CR3BP_f, CR3BP_df

def CR3BP_Lyap_ICs(Lagrn_pt: np.array, mu: float, amp: float=1e-2) -> tuple[np.array,float]:
    """ Computes ballpark initial conditions for a planar orbit around a Lagrange point via linearizing the EOMs.

    Parameters:
    Lagrn_pt (np.array): Position of the Lagrange point [x, y, z].
    mu (float): Mass ratio of the two primary bodies.
    amp (float): Amplitude of the initial perturbation.

    Returns:
    np.array: Initial state vector [x, y, z, vx, vy, vz] for the Lyapunov orbit.
    """
    # Compute the Jacobian of the EOMs at the Lagrange point
    X_eq = np.array([Lagrn_pt[0], Lagrn_pt[1], Lagrn_pt[2], 0.0, 0.0, 0.0])
    A = CR3BP_dfdX(X_eq, mu)
    
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
    dX0 = v*amp  # required perturbation to ICs
    T = 2 * np.pi / np.abs(w[idx_lyap[0]].imag)  # period of the linearized motion
    
    return dX0, T

def CR3BP_Jacobi(X: np.array, mu: float) -> float:
    """ Computes the Jacobi constant for a given state in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    X (np.array): State vector [x, y, z, vx, vy, vz].
    mu (float): Mass ratio of the two primary bodies.

    Returns:
    float: The Jacobi constant.
    """
    r = X[:3]
    v = X[3:]
    
    U = CR3BP_U(r, mu)
    v2 = np.dot(v, v)
    
    JC = 2 * U - v2
    return JC

def CR3BP_BrkVals(Phi: np.array) -> np.array:
    """ Computes the stability indices (BrkVals) for a periodic orbit in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    Phi (np.array): State transition matrix (STM) at the end of the period.
    mu (float): Mass ratio of the two primary bodies.

    Returns:
    np.array: Array containing the two stability indices [alpha, beta].
    """
    alpha = 2-np.trace(Phi)
    beta = 0.5*(alpha**2 + 2 - np.trace(Phi @ Phi))

    return np.array([alpha, beta])

def CR3BP_PseudArcL(dfdfv: np.array, cont: dict = None) -> np.array:
    """ Computes the pseudo-arclength tangent vector for continuation of periodic orbits in the Circular Restricted Three-Body Problem (CR3BP).

    Parameters:
    dfdfv (np.array): Jacobian of the residual with respect to the free variables of current orbit
    cont (dict): Continuation parameters
        - 'type' (str): Type of continuation ('family' or 'bifurcation').
        - 'norm_dst' (float): Desired step size in the continuation.
        - 'align_vec' (np.array): Tangent vector from the previous step for continuity.
        - 'norm_dims' (np.array): Indices of the dimensions to normalize the step size.

    Returns:
    np.array: The pseudo-arclength tangent vector.
    """
    # Perform SVD on the augmented matrix
    U, S, Vh = svd(dfdfv)
    V = Vh.conj().T
    
    # Determine the tangent vector
    if cont['type'] == 'family':
        # Take the singular vector corresponding to the second smallest singular value
        tangent = V[:, -1]
    elif cont['type'] == 'bifurcation':
        tangent = V[:, -2]
        
    # Align the tangent with align_vec
    if np.dot(tangent, cont['align_vec']) < 0:
        tangent = -tangent  # flip direction for continuity
    # Normalize the tangent vector based on specified dimensions
    tangent = (tangent / norm(tangent[cont['norm_dims']])) * cont["norm_dst"]

    return tangent

def CR3BP_Bifur_Detec(X0s: np.array, Ts: float, Brks: np.array, Type: Literal["Tan","P2","P3","P4"]) -> tuple[np.array,np.array]:
    """ This function detects bifurcations of a given type

    Parameters:
    X0s (np.array): Initial states before and after bifurcation
    Ts (float): Orbit period ...
    Brks (np.array): Brouke values ...
    Type (string): Bifurcation type

    Returns: 
    X0_b (np.array): Approximated X0 at bifurcation
    T_b (np.array): Approximated T at bifurcation
    """
    if Type == "Tan":
        beta = lambda alpha: -2*alpha - 2
    elif Type == "P2":
        beta = lambda alpha: 2*alpha - 2
    elif Type == "P3":
        beta = lambda alpha: alpha + 1
    elif Type == "P4":
        beta = lambda alpha: 2
    
        # set up linear interpolation points
    orb_P1 = Brks[0,:]
    orb_P2 = Brks[1,:]

    bif_P1 = np.array([Brks[0,0],beta(Brks[0,0])])
    bif_P2 = np.array([Brks[1,0],beta(Brks[1,0])])

    # Find the approximate intersection point
    a11 = det(np.array([orb_P1,
                        orb_P2]))
    a12 = det(np.array([[orb_P1[0], 1],
                        [orb_P2[0], 1]]))
    a21 = det(np.array([bif_P1, 
                        bif_P2]))
    a22 = det(np.array([[bif_P1[0], 1],
                        [bif_P2[0], 1]]))
    
    b12 = det(np.array([[orb_P1[1], 1],
                        [orb_P2[1], 1]]))
    b22 = det(np.array([[bif_P1[1], 1],
                        [bif_P2[1], 1]]))
    
    num = det(np.array([[a11, a12], 
                        [a21, a22]]))
    den = det(np.array([[a12, b12],
                        [a22, b22]]))
    
    # Brouke alpha value at intersection
    alph_bif = num/den

    # Bifurcation distance between the pre and post values
    d = (alph_bif-Brks[0,0])/(Brks[1,0]-Brks[0,0])

    X0_b = (1-d)*X0s[0,:] + d*X0s[1,:]
    T_b = (1-d)*Ts[0] + d*Ts[1]

    return X0_b, T_b

def fix_phase(v: np.array) -> np.array:
    """ Fixes the phase of a vector v by normalizing it based on its first non-zero element.

    Parameters:
    v (np.array): The vector whose phase needs to be fixed.

    Returns:
    np.array: The phase-fixed vector.
    """
    k = np.flatnonzero(np.abs(v) > 0)
    if k.size:
        a = v[k[0]]
        v = v * (a.conjugate() / abs(a))
    return v

def fix_phase_all(V: np.array) -> np.array:
    """Apply fix_phase to every eigenvector (column) independently.

    Parameters:
    V (np.array): Matrix whose columns are eigenvectors.

    Returns:
    np.array: Matrix with phase-fixed eigenvectors as columns.
    """

    W = V.copy()
    for j in range(W.shape[1]):
        W[:, j] = fix_phase(W[:, j])
    return W

def pair_reciprocals(w: np.array, v: np.array, tol_recip: float=1e-8, tol_unit: float=1e-8) -> tuple[np.array, np.array]:
    """ Pairs eigenvalues and eigenvectors that are reciprocals of each other
    
    Parameters:
    w (np.array): Array of eigenvalues.
    v (np.array): Matrix whose columns are eigenvectors.
    tol_recip (float): Tolerance for determining if two eigenvalues are reciprocals.
    tol_unit (float): Tolerance for determining if an eigenvalue is on the unit circle.

    Returns:
    tuple: A tuple containing:
        - w_paired (np.array): Eigenvalues paired as reciprocals.
        - v_paired (np.array): Corresponding eigenvectors.
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

def eig_sort(A: np.array,prev_vecs: np.array = None, tol_one: float=1e-4, tol_unit: float=1e-4, tol_recip: float=1e-8) -> tuple[np.array, np.array]:
    """ Sorts the eigenvalues and eigenvectors of a matrix A to maintain continuity with previous eigenvectors. 
    Following rules are established if no previous basis is given:
    1. Vals and Vecs paired as reciprocals
    2. Order by |λ|≈1 first, then angle off real.
    3. Order by real vals next, then complex.
    If previous basis is given, the new basis is chosen to maximize overlap with previous basis.
    
    Parameters:
    A (np.array): The input matrix.
    prev_vecs (np.array): The previous eigenvectors for continuity.

    Returns:
    tuple: A tuple containing:
        - w_out (np.array): Sorted eigenvalues.
        - V_out (np.array): Sorted eigenvectors.
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

def save_family(family_data: dict,output_loc: str):    
    with h5py.File(output_loc, "w") as f:
        f.create_dataset("X_hst",data=family_data["X_hst"],compression="gzip",chunks=True)
        f.create_dataset("t_hst",data=family_data["t_hst"],compression="gzip",chunks=True)
        f.create_dataset("JCs",data=family_data["JCs"],compression="gzip",chunks=True)
        f.create_dataset("STMs",data=family_data["STMs"],compression="gzip",chunks=True)
        f.create_dataset("BrkVals",data=family_data["BrkVals"],compression="gzip",chunks=True)

def load_family(input_loc: str) -> dict:
    family_data = {}
    with h5py.File(input_loc, "r") as f:
        family_data["X_hst"]   = f["X_hst"][:]     # (6 x n x N) array
        family_data["t_hst"]   = f["t_hst"][:]     # (1 x n x N) array
        family_data["JCs"]     = f["JCs"][:]       # (1 x N) array
        family_data["STMs"]    = f["STMs"][:]      # (6 x 6 x N) array
        family_data["BrkVals"] = f["BrkVals"][:]   # (whatever shape you saved)

    return family_data