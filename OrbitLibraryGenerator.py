# OrbitLibraryGenerator.py
# Generates orbit libraries for the CR3BP system based on parameters defined in the .yaml file.

import yaml
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve
import jax
import jax.numpy as jnp
import diffrax as dfx

# JAX precision
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from util import CR3BP_symOrb_solver

path = Path(__file__).parent / "EarthMoon System" / "EMsys.yaml"  # Change this to the desired system name
with open(path, "r") as file:  
    Sys = yaml.safe_load(file)

mu = Sys['mu']

fv_guess = np.array([0.8,0,0.346,1.65])

fv_sol = CR3BP_symOrb_solver(fv_guess, mu, fixed_var="free", ftol=1e-12)