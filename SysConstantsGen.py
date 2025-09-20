# SysConstantsGen.py
# Generates system constants for the CR3BP system based on parameters defined in the .yaml file.

import yaml
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve

G_dim =  6.67430e-20  # Gravitational constant in km^3/kg/s^2 (https://ssd.jpl.nasa.gov/astro_par.html)

out_path = Path(__file__).parent / "EarthMoon System/EMSys.yaml" # Change this to the desired system name


params_path = Path(__file__).parent / "SystemParameters.yaml"
with open(params_path, "r") as file:
    params = yaml.safe_load(file)


# Extract parameters
primary_GM = params['primary_GM']  # GM of Primary
secondary_GM = params['secondary_GM']  # GM of Secondary
distance = params['distance']  # Distance between primary and secondary bodies in km

# System masses in kg
m1 = primary_GM / G_dim  # Mass of primary in kg
m2 = secondary_GM / G_dim  # Mass of secondary in kg

# Generate system constants dictionary
Sys = {
    'dim': {
        'm1': m1,  # Mass of primary in kg
        'm2': m2,  # Mass of secondary in kg
        'l': distance,  # Distance between primary and secondary in km
    },
    'mu': m2/(m1+m2),  # Mass ratio
    'Ts': (distance**3/(G_dim*(m1+m2))),  # Characteristic time in seconds
    'Ms': m1+m2,  # Characteristic mass (total mass of the system) in kg
    'Ls': distance,  # Characteristic length (distance between primary and secondary) in km 
}

Sys['Vs'] = Sys['Ls']/Sys['Ts']  # Characteristic velocity in km/s
Sys['As'] = Sys['Vs']/Sys['Ts']  # Characteristic acceleration in km/s^2



# Calculate Lagrange points
mu = Sys['mu']
def dUx(x, mu): # Derivative of the pseudo-potential function U with respect to x
    r1 = np.abs(x+mu)
    r2 = np.abs(x-1+mu)
    return x - (1 - mu)*(x + mu)/(r1**3) - mu*(x - 1 + mu)/(r2**3)

Sys['LagrPts'] = {}
Sys['LagrPts']['L1'] = np.asarray([fsolve(lambda x: dUx(x, mu), 1 - mu - (mu/3)**(1/3), xtol=1e-12)[0],0,0]).tolist()
Sys['LagrPts']['L2'] = np.asarray([fsolve(lambda x: dUx(x, mu), 1 - mu + (mu/3)**(1/3), xtol=1e-12)[0],0,0]).tolist()
Sys['LagrPts']['L3'] = np.asarray([fsolve(lambda x: dUx(x, mu), -1 - 5*mu/12, xtol=1e-12)[0],0,0]).tolist()
Sys['LagrPts']['L4'] = np.asarray([0.5-mu, np.sqrt(3)/2,0]).tolist()
Sys['LagrPts']['L5'] = np.asarray([0.5-mu, -np.sqrt(3)/2,0]).tolist()

# Save System Data to YAML file
with open(out_path, "w") as file:
    yaml.safe_dump(dict(Sys), file, sort_keys=False)



