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
from util import CR3BP_symOrb_solver, CR3BP_Phi_jax, CR3BP_dPhidX0_jax, eig_sort, CR3BP_Lyap_ICs, CR3BP_PseudArcL, CR3BP_Traj_Sol, CR3BP_Jacobi, CR3BP_BrkVals, CR3BP_symOrb_f, CR3BP_symOrb_df, save_family

path = Path(__file__).parent / "EarthMoon System" / "EMsys.yaml"  # Change this to the desired system name
with open(path, "r") as file:  
    Sys = yaml.safe_load(file)

mu = Sys['mu']

# Family continuation 
Norbs = 250
Npts = 2000
name = "L1_Lyap"
save_path = Path(__file__).parent / "EarthMoon System" / name + ".h5"

# Initialize storage arrays
Family_data = {"Name": name, 
               "X_hst": np.zeros((Norbs,Npts,6)),
               "t_hst": np.zeros((Norbs,Npts)), 
               "JCs": np.zeros(Norbs), 
               "STMs": np.zeros((Norbs,6,6)), 
               "BrkVals": np.zeros((Norbs,2))}

# State to free variable and vice versa transformation functions
def X2fv(X, T): return np.array([X[0], X[2], X[4], T/2])
def fv2X_T(fv): return np.array([fv[0], 0, fv[1], 0, fv[2], 0]), fv[3]*2

# Initial orbit
print("Computing initial orbit...") 
X0,T = CR3BP_Lyap_ICs(Sys['LagrPts']['L1'], mu, amp=-1e-4)
fv_guess = X2fv(X0,T)
sol = CR3BP_symOrb_solver(fv_guess, mu, fixed_var="x")
fv_sol = sol["x"]
dfdfv_sol = CR3BP_symOrb_df(fv_sol, mu,"free")
X0,T = fv2X_T(fv_sol)

Family_data["X_hst"][0,:,:], Family_data["t_hst"][0,:] = CR3BP_Traj_Sol(X0, T, mu, Npts)
Family_data["JCs"][0] = CR3BP_Jacobi(X0, mu)
Family_data["STMs"][0,:,:] = CR3BP_dPhidX0_jax(X0, T, mu)
Family_data["BrkVals"][0,:] = CR3BP_BrkVals(Family_data["STMs"][1,:,:])


# Continuation loop
for k in range(1,Norbs):
    print(f"Computing orbit {k+1:3d} of {Norbs}...    ", end="")
    IC_step_dst = 1e-3 # Step distance between ICs
    norm_dims = np.array([0])

    # Set the alignment vector for pseudo-arclength continuation
    if k == 1:
        align_vec = np.array([-1,0,0,0])
    else:
        align_vec = dfv

    # Continuation iteration 
    continuation = {"type": "family", "norm_dst": IC_step_dst, "align_vec": align_vec, "norm_dims": norm_dims} # Continuation parameters
    dfv = CR3BP_PseudArcL(dfdfv_sol, continuation) # Estimate the continuation direction
    fv_guess = fv_sol + dfv # Generate new free variable guess
    init_cost = 0.5*norm(CR3BP_symOrb_f(fv_guess, mu)) # Print Intial Cost
    sol = CR3BP_symOrb_solver(fv_guess, mu, fixed_var="x") # Solve for the orbit
    fv_sol = sol["x"] # Extract free variables
    dfdfv_sol = CR3BP_symOrb_df(fv_sol, mu,"free") # Set up jacobian for estimating continuation direction
    print(f"OutFlag {sol["status"]}, Cost_0: {init_cost:.2e}, Cost_F: {sol["cost"]:.2e},    f_eval: {sol["nfev"]:2d}")
    X0,T = fv2X_T(fv_sol)

    # Propagate orbit and save to data matrices
    Family_data["X_hst"][k,:,:], Family_data["t_hst"][k,:] = CR3BP_Traj_Sol(X0, T, mu, Npts = Npts)
    Family_data["JCs"][k] = CR3BP_Jacobi(X0, mu)
    Family_data["STMs"][k,:,:] = CR3BP_dPhidX0_jax(X0, T, mu)
    Family_data["BrkVals"][k,:] = CR3BP_BrkVals(Family_data["STMs"][k,:,:])



# Save data
#save_family(Family_data,save_path)