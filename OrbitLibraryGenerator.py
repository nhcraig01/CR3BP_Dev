# OrbitLibraryGenerator.py
# Generates orbit libraries for the CR3BP system based on parameters defined in the .yaml file.

import yaml
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve
import jax
import jax.numpy as jnp
import diffrax as dfx
import importlib
import h5py


# JAX precision
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import util
from util import CR3BP_symOrb_solver, CR3BP_Phi_jax, CR3BP_dPhidX0_jax, eig_sort, CR3BP_Lyap_ICs
importlib.reload(util)

path = Path(__file__).parent / "EarthMoon System" / "EMsys.yaml"  # Change this to the desired system name
with open(path, "r") as file:  
    Sys = yaml.safe_load(file)

mu = Sys['mu']

"""
fv_guess = np.array([0.8,0,0.346,1.65])

fv_sol = CR3BP_symOrb_solver(fv_guess, mu, fixed_var="free")

print("fv_sol:", fv_sol)

X0 = np.array([fv_sol[0],0,fv_sol[1],0,fv_sol[2],0])
T = fv_sol[3]*2

Xf = CR3BP_Phi_jax(X0, T, mu)
dPhidX0 = CR3BP_dPhidX0_jax(X0, T, mu)

Xerr = Xf - X0


print("X0:", X0)
print("T:", T)
print("Xerr:", Xerr)
"""
X0, T = CR3BP_Lyap_ICs(Sys['LagrPts']['L1'], mu, amp=-1e-2)
fv_guess = np.array([X0[0], X0[2], X0[4], T/2])
print("fv_guess (Lyap):", fv_guess)
fv_sol = CR3BP_symOrb_solver(fv_guess, mu, fixed_var="free")
print("fv_sol (Lyap):", fv_sol)


# Family continuation
Norbs = 250
Npts = 2000
family_name = "L1_Lyap.h5"

X_hst = np.zeros((Norbs,Npts,6))
T_hst = np.zeros((Norbs,Npts))
JCs = np.zeros(Norbs)
Phis = np.zeros((Norbs,6,6))
BrkVals = np.zeros((Norbs,2))

# Start here, just starting the orbit family generation.
# 1. Make function to SVD and do pseudo-arclength continuation
# 2. Intialize loop to generate orbits
# 3. Loop through and save orbits to h5 file
# 4. Go and plot in matlab


with h5py.File(path.parent / family_name, "w") as f:
    f.create_dataset("X_hst",data=np.zeros((Norbs,Npts,6)),compression="gzip",chunks=True)
    f.create_dataset("T_hst",data=np.zeros((Norbs,Npts)),compression="gzip",chunks=True)
    f.create_dataset("JCs",data=np.zeros(Norbs),compression="gzip",chunks=True)
    f.create_dataset("Phi",data=np.zeros((Norbs,6,6)),compression="gzip",chunks=True)
    f.create_dataset("BrkVals",data=np.zeros(Norbs,2),compression="gzip",chunks=True)
