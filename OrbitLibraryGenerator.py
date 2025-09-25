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
from util import CR3BP_Phi_jax, CR3BP_dPhidX0_jax, eig_sort, CR3BP_Lyap_ICs, CR3BP_PseudArcL, CR3BP_Traj_Sol, CR3BP_Jacobi, CR3BP_BrkVals, save_family, load_family, CR3BP_Orb_solver_funcs, CR3BP_Bifur_Detec

# Intialize system parameters
path = Path(__file__).parent / "EarthMoon System" / "EMsys.yaml"  # Change this to the desired system name
with open(path, "r") as file:  
    Sys = yaml.safe_load(file)
mu = Sys['mu']

# Family continuation parameters
Norbs = 200
Npts = 2000
name = 'L2_Lyap' # family name
solver_type = "Symmetric" # type of orbit solver to be used
IC_step_dst = 1e-3 # Step distance between ICs
align_vec0 = [1,0,0,0] # intial vector to move along for family continuation
norm_dims = np.array([0]) # dimensions of free variable to normalize against
fixed_var0 = "x" # solution initial fixed variable

# Generate first orbit from linearizing about Lagrange point
init_orb = {"type": "Lp_Lin", "Lp": "L2", "Step_off_dist": 1e-4}

# Generate first orbit from bifurcation in another family
# bifur_family_path = Path(__file__).parent / "EarthMoon System" / "L1_Lyap.h5"
# init_orb = {"type": "Bifur", "Bifur_type": "Tan", "indxs": [14,15], "bifur_family_path": bifur_family_path}


# Data save path
save_path = Path(__file__).parent / "EarthMoon System" / (name+".h5")



# Genrate family, relatrively automated
# Initialize storage arrays
Family_data = {"X_hst": np.zeros((Norbs,Npts,6)),
               "t_hst": np.zeros((Norbs,Npts)), 
               "JCs": np.zeros(Norbs), 
               "STMs": np.zeros((Norbs,6,6)), 
               "BrkVals": np.zeros((Norbs,2))}

# Solver functions
X2fv, fv2X_T, CR3BP_Orb_solver, CR3BP_Orb_f, CR3BP_Orb_df = CR3BP_Orb_solver_funcs(solver_type)

# Initial orbit
print("Computing initial orbit ",end="") 
if init_orb["type"] == "Lp_Lin":
    print("via Lagrange point linearization")
    dX0,T = CR3BP_Lyap_ICs(Sys['LagrPts'][init_orb['Lp']], mu, amp=init_orb["Step_of_dist"])
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

    fv_b_guess = X2fv(X0_b, T_b)
    sol = CR3BP_Orb_solver(fv_b_guess, mu, fixed_var0)
    fv_b = sol["x"]
    dfdfv_b = CR3BP_Orb_df(fv_b,mu)
    continuation = {"type": "bifurcation", "norm_dst": IC_step_dst, "align_vec": align_vec0, "norm_dims": norm_dims} # Continuation parameters
    dfv = CR3BP_PseudArcL(dfdfv_b, continuation)
    fv_guess = fv_b+dfv

sol = CR3BP_Orb_solver(fv_guess, mu, fixed_var=fixed_var0)
fv_sol = sol["x"]
dfdfv_sol = CR3BP_Orb_df(fv_sol, mu,"free")
X0,T = fv2X_T(fv_sol)

# Allocate space for data
Family_data["X_hst"][0,:,:], Family_data["t_hst"][0,:] = CR3BP_Traj_Sol(X0, T, mu, Npts)
Family_data["JCs"][0] = CR3BP_Jacobi(X0, mu)
Family_data["STMs"][0,:,:] = CR3BP_dPhidX0_jax(X0, T, mu)
Family_data["BrkVals"][0,:] = CR3BP_BrkVals(Family_data["STMs"][0,:,:])


# Continuation loop
for k in range(1,Norbs):
    print(f"Computing orbit {k+1:3d} of {Norbs}...    ", end="")

    # Set the alignment vector for pseudo-arclength continuation
    if 'dfv': align_vec = dfv
    else: align_vec = align_vec0

    # Continuation iteration 
    continuation = {"type": "family", "norm_dst": IC_step_dst, "align_vec": align_vec, "norm_dims": norm_dims} # Continuation parameters
    dfv = CR3BP_PseudArcL(dfdfv_sol, continuation) # Estimate the continuation direction
    fv_guess = fv_sol + dfv # Generate new free variable guess
    init_cost = 0.5*norm(CR3BP_Orb_f(fv_guess, mu)) # Print Intial Cost
    sol = CR3BP_Orb_solver(fv_guess, mu) # Solve for the orbit
    fv_sol = sol["x"] # Extract free variables
    dfdfv_sol = CR3BP_Orb_df(fv_sol, mu,"free") # Set up jacobian for estimating continuation direction
    print(f"OutFlag {sol["status"]}, Cost_0: {init_cost:.2e}, Cost_F: {sol["cost"]:.2e},    f_eval: {sol["nfev"]:2d}")
    X0,T = fv2X_T(fv_sol)

    # Propagate orbit and save to data matrices
    Family_data["X_hst"][k,:,:], Family_data["t_hst"][k,:] = CR3BP_Traj_Sol(X0, T, mu, Npts = Npts)
    Family_data["JCs"][k] = CR3BP_Jacobi(X0, mu)
    Family_data["STMs"][k,:,:] = CR3BP_dPhidX0_jax(X0, T, mu)
    Family_data["BrkVals"][k,:] = CR3BP_BrkVals(Family_data["STMs"][k,:,:])



# Save data
save_family(Family_data,save_path)