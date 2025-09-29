# OrbitLibraryGenerator.py
# Generates orbit libraries for the CR3BP system based on parameters defined in the .yaml file.

import yaml
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy.optimize import check_grad
import jax
import jax.numpy as jnp
import diffrax as dfx
import h5py

# JAX precision
jax.config.update("jax_enable_x64", True)

import util
from util import CR3BP_Phi_jax, CR3BP_dPhidX0_jax, eig_sort, CR3BP_Lyap_ICs, CR3BP_PseudArcL, CR3BP_Traj_Sol, CR3BP_Jacobi, CR3BP_BrkVals, save_family, load_family, CR3BP_Orb_solver_funcs, CR3BP_Bifur_Detec, CR3BP_Zplane_Cross_ICs

# Intialize system parameters
sys_folder = "EarthMoon System"
path = Path(__file__).parent / sys_folder / "EMsys.yaml"  # Change this to the desired system name
with open(path, "r") as file:  
    Sys = yaml.safe_load(file)
mu = Sys['mu']

# Family continuation parameters (Edit this section to generate different families)
Norbs = 10
Npts = 2000
name = 'idky'                  # family name
solver_type = "Asymmetric"           # type of orbit solver to be used
z0 = 0
z0_dir = -1
IC_step_dst0 = 1e-3                 # Initial step off distance from 
IC_step_dst = 4e-3                  # Step distance between ICs
align_vec0 = [0,1,0,0,0,0]          # intial vector to move along for family continuation
norm_dims = np.array([0,1])         # dimensions of free variable to normalize against
fixed_var = "x"
fixed_var0 = "free"                    # solution initial fixed variable

# Generate first orbit from linearizing about Lagrange point
# init_orb = {"type": "Lp_Lin", "Lp": "L3", "Step_off_dist": IC_step_dst0*align_vec0[0]}

# Generate first orbit from bifurcation in another family
bifur_family_path = Path(__file__).parent / sys_folder / "L1_N_Halo.h5"
init_orb = {"type": "Bifur", "Bifur_type": "Tan", "indxs": [251,252], "bifur_family_path": bifur_family_path}





# Genrate family, relatrively automated
# Initialize storage arrays
Family_data = {"X_hst": np.zeros((Norbs,Npts,6)),
               "t_hst": np.zeros((Norbs,Npts)), 
               "JCs": np.zeros(Norbs), 
               "STMs": np.zeros((Norbs,6,6)), 
               "BrkVals": np.zeros((Norbs,2))}

# Solver functions
X2fv, fv2X_T, CR3BP_Orb_solver, CR3BP_Orb_f, CR3BP_Orb_df = CR3BP_Orb_solver_funcs(solver_type,z0)

# Jacobian Jax vs Finite Diff Check
"""
def phi_np(X0):
    X0 = jnp.asarray(X0)
    Xf = CR3BP_Orb_f(X0, mu = mu, rtol=1e-12, atol=1e-12)
    return np.asarray(Xf)

def J_np(X0):
    X0 = jnp.asarray(X0)
    J = CR3BP_Orb_df(X0, mu = mu, rtol=1e-12, atol=1e-12)
    return np.asarray(J)

def fd_jacobian(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    y0 = f(x)
    m, n = y0.size, x.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        e = np.zeros_like(x); e[j] = 1.0
        J[:, j] = (f(x + eps*e) - f(x - eps*e)) / (2*eps)
    return J

# pick a representative X0
X0 = np.array([0.8, 0, 0, 0.1, 0, 2], dtype=float)
# X0 = np.array([0.8,0,0.2,1])

J_ad = J_np(X0)
J_fd = fd_jacobian(phi_np, X0, eps=1e-6)

# relative matrix error (Frobenius)
num = np.linalg.norm(J_ad - J_fd, ord='fro')
den = max(1.0, np.linalg.norm(J_fd, ord='fro'))
print("J_ad: ", J_ad)
print("J_fd: ", J_fd)
print("Jacobian relative error:", num/den)
"""


# Initial orbit
print("Initial orbit ",end="") 
if init_orb["type"] == "Lp_Lin":
    print("via Lagrange point linearization")
    dX0,T = CR3BP_Lyap_ICs(Sys['LagrPts'][init_orb['Lp']], mu, amp=init_orb["Step_off_dist"])
    X0 = np.concat([Sys['LagrPts'][init_orb['Lp']],np.zeros(3)])
    fv = X2fv(X0,0)
    dfv = X2fv(dX0,T)
    fv_guess = fv+dfv
elif init_orb["type"] == "Bifur":
    print("via bifurcation pseudo-arc length")
    Bifur_family = load_family(init_orb["bifur_family_path"])
    X0s_bif = Bifur_family["X_hst"][init_orb["indxs"],0,:]
    Ts_bif = Bifur_family["t_hst"][init_orb["indxs"],-1]
    Brks_bif = Bifur_family["BrkVals"][init_orb["indxs"],:]
    Type_bif = init_orb["Bifur_type"]
    X0_b, T_b = CR3BP_Bifur_Detec(X0s_bif, Ts_bif, Brks_bif, Type_bif)
    if solver_type == "Asymmetric":
        X0_b = CR3BP_Zplane_Cross_ICs(X0_b, T_b, mu, z0, z0_dir)

    fv_b_guess = X2fv(X0_b, T_b)
    # print(0.5*norm(CR3BP_Orb_f(fv_b_guess, mu=mu)))

    sol = CR3BP_Orb_solver(fv_b_guess, mu = mu, fixed_var="free")

    fv_b = sol["x"]
    dfdfv_b = CR3BP_Orb_df(fv_b,mu = mu)
    continuation = {"type": "bifurcation", "norm_dst": IC_step_dst0, "align_vec": align_vec0, "norm_dims": norm_dims} # Continuation parameters
    dfv = CR3BP_PseudArcL(dfdfv_b, continuation)
    # print(dfdfv_b@dfv)
    fv_guess = fv_b+dfv

print(f"Computing orbit {1:3d} of {Norbs}...    ", end="")
init_cost = 0.5*norm(CR3BP_Orb_f(fv_guess, mu = mu)) # Print Intial Cost
sol = CR3BP_Orb_solver(fv_guess, mu = mu, fixed_var=fixed_var0)
fv_sol = sol["x"]
dfdfv_sol = CR3BP_Orb_df(fv_sol, mu = mu, fixed_var="free")
X0,T = fv2X_T(fv_sol)
print(f"OutFlag {sol["status"]}, Cost_0: {init_cost:.1e}, Cost_F: {sol["cost"]:.1e},    f_eval: {sol["nfev"]:2d}")

# Allocate space for data
Family_data["t_hst"][0,:] = np.linspace(0,T,Npts)
Family_data["X_hst"][0,:,:] = CR3BP_Phi_jax(X0,T,mu,t_hst_out = Family_data["t_hst"][0,:])
Family_data["JCs"][0] = CR3BP_Jacobi(X0, mu)
Family_data["STMs"][0,:,:] = CR3BP_dPhidX0_jax(X0, T, mu)
Family_data["BrkVals"][0,:] = CR3BP_BrkVals(Family_data["STMs"][0,:,:])

# Continuation loop
for k in range(1,Norbs):
    print(f"Computing orbit {k+1:3d} of {Norbs}...    ", end="")

    # Set the alignment vector for pseudo-arclength continuation
    align_vec = dfv

    # Continuation iteration 
    continuation = {"type": "family", "norm_dst": IC_step_dst, "align_vec": align_vec, "norm_dims": norm_dims} # Continuation parameters
    dfv = CR3BP_PseudArcL(dfdfv_sol, continuation) # Estimate the continuation direction
    fv_guess = fv_sol + dfv # Generate new free variable guess
    init_cost = 0.5*norm(CR3BP_Orb_f(fv_guess, mu = mu)) # Print Intial Cost
    sol = CR3BP_Orb_solver(fv_guess, mu = mu, fixed_var=fixed_var) # Solve for the orbit
    fv_sol = sol["x"] # Extract free variables
    dfdfv_sol = CR3BP_Orb_df(fv_sol, mu = mu, fixed_var="free") # Set up jacobian for estimating continuation direction
    X0,T = fv2X_T(fv_sol) # Extract values

    # Print solution statement
    print(f"OutFlag {sol["status"]}, Cost_0: {init_cost:.1e}, Cost_F: {sol["cost"]:.1e},    f_eval: {sol["nfev"]:2d}")

    # Propagate orbit and save to data matrices
    Family_data["t_hst"][k,:] = np.linspace(0,T,Npts)
    Family_data["X_hst"][k,:,:] = CR3BP_Phi_jax(X0,T,mu,t_hst_out = Family_data["t_hst"][k,:])
    Family_data["JCs"][k] = CR3BP_Jacobi(X0, mu)
    Family_data["STMs"][k,:,:] = CR3BP_dPhidX0_jax(X0, T, mu)
    Family_data["BrkVals"][k,:] = CR3BP_BrkVals(Family_data["STMs"][k,:,:])



# Save data
save_path = Path(__file__).parent / sys_folder / (name+".h5")
save_family(Family_data,save_path)