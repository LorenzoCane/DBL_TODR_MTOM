
import intake
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import inspect
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares

from openap import prop #aircraft and engine-related data
from openap.drag import Drag # drag related
from openap.kinematic import WRAP #set of kinematic models
from openap.thrust import Thrust #thrust calc

from pprint import pprint #“pretty-print” arbitrary Python data structures 

from utils import ComplexUnitConverter as conv
from utils import rmsd
from take_off_func import take_off, cl_finder


#***************************************************************************
#constants

full_mass = 78000.0 #kg
brussels_lenght = 3638.0 #m
brussel_alt = conv.convert(56.0, 'm', 'ft') #m
cl_best = 1.641

r_spec = 287.0 # (N*m) / (kg*K) 
pathway_incl = 0.0 #deg

asc = conv.convert(35.0, 'ft', 'm') # m
climb_ang = 7.7 # deg (see [Gratton et al. 2020])

airborne_dist = asc / np.tan(conv.convert(climb_ang, 'deg', 'rad')) # m
#print(airborne_dist)

safe_margin_coef = 1.1

os.makedirs('images', exist_ok=True)
img_dir = './images/'

input_dir = './data/clean'
#***************************************************************************
#aircraft
aircraft_name = "A320"
engine_name = "V2500-A1"

#available_aircraft = prop.available_aircraft() #list of avaib aircraft if needed

aircraft = prop.aircraft(aircraft_name) #airbus A320
#pprint(aircraft)
engine = prop.engine(engine_name) #V2500-A1 turbofan engines
wing_area = aircraft['wing']['area'] #wing area
cd0 = aircraft['drag']['cd0']
k = aircraft['drag']['k']
mu = 0.017
#print(k)
wrap = WRAP(ac=aircraft_name) #kinematic parameters

to_speed = wrap.takeoff_speed() # m/s Take-off speed. order: default (optimum), minimum, maximum
#print(to_speed)

opt_to_speed = to_speed['default'] #m/s
min_to_speed = to_speed['minimum'] #m/s
max_to_speed = to_speed['maximum'] #m/s
speed_val = np.sort([s for s in list(to_speed.values())[:3]])
print(speed_val)

thr_a320 = Thrust(ac= aircraft_name, eng= engine_name)
T = np.array([thr_a320.takeoff(tas = conv.convert(i, 'ms', 'kts'), alt=brussel_alt) for i in speed_val]) #N

#***************************************************************************

file_dict = {
    "Historical": "Historical_JJA.csv",
    "SSP126": "SSP126_JJA.csv",
    "SSP370": "SSP370_JJA.csv",
    "SSP585": "SSP585_JJA.csv"
}

img_name = 'Brussels_NOQDM.pdf'

all_data = []  # list to hold each scenario's processed DataFrame
for scenario, filename in file_dict.items():
    # Read the CSV; each file should have at least columns: "mx2t24" (temperature, [K]) and "sp" (pressure, [Pa])
    df = pd.read_csv(os.path.join(input_dir, filename))
    
    # Compute air density: rho = Pressure / (R * Temperature)
    df["rho"] = df["sp"] / (r_spec * df["mx2t24"])
    
    # Compute TODR for each row using the same constant parameters
    df["TODR"] = df.apply(lambda row: take_off(full_mass, T[1], row["rho"], cl_best, cd0, k, wing_area, airborne_dist, safe_margin_coef, mu, pathway_incl), axis=1)
    
    # Add a column for the scenario label
    df["Scenario"] = scenario
    
    # Remove rows where TODR, temperature, or pressure are NaN
    df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    all_data.append(df)

# Concatenate all the scenario DataFrames
df_all = pd.concat(all_data, ignore_index=True)
print("\n=== TODR Summary by Scenario ===")
print(df_all.groupby("Scenario")["TODR"].describe().round(2))

# --- Create the Boxplot ---
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# Define custom order and colors for scenarios if desired
order = ["Historical", "SSP126", "SSP370", "SSP585"]
palette = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]

sns.boxplot(x="Scenario", y="TODR", data=df_all, whis =1.5, order=order, palette=palette, showfliers=True)
plt.xlabel("Scenario")
plt.ylabel("TODR [m]")
plt.title("Computed Take-Off Distance Required (TODR)\n(JJA Data)")
#sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, img_name))



#***************************************************************************
#with QDM
file_dict = {
    "Historical": "Historical_QDM_JJA.csv",
    "SSP126": "SSP126_QDM_JJA.csv",
    "SSP370": "SSP370_QDM_JJA.csv",
    "SSP585": "SSP585_QDM_JJA.csv"
}

img_name = 'Brussels_QDM.pdf'

all_data = []  # list to hold each scenario's processed DataFrame
for scenario, filename in file_dict.items():
    # Read the CSV; each file should have at least columns: "mx2t24" (temperature, [K]) and "sp" (pressure, [Pa])
    df = pd.read_csv(os.path.join(input_dir, filename))
    
    # Compute air density: rho = Pressure / (R * Temperature)
    df["rho"] = df["sp"] / (r_spec * df["mx2t24"])
    
    # Compute TODR for each row using the same constant parameters
    df["TODR"] = df.apply(lambda row: take_off(full_mass, T[1], row["rho"], cl_best, cd0, k, wing_area, airborne_dist, safe_margin_coef, mu, pathway_incl), axis=1)
    
    # Add a column for the scenario label
    df["Scenario"] = scenario
    
    # Remove rows where TODR, temperature, or pressure are NaN
    df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    all_data.append(df)

# Concatenate all the scenario DataFrames
df_all = pd.concat(all_data, ignore_index=True)

# --- Create the Boxplot ---
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# Define custom order and colors for scenarios if desired
order = ["Historical", "SSP126", "SSP370", "SSP585"]
palette = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]

sns.boxplot(x="Scenario", y="TODR", data=df_all,whis=1.5, order=order, palette=palette, showfliers=True)
plt.ylim(2090, 2350)
plt.xlabel("Scenario")
plt.ylabel("TODR [m]")
plt.title("Computed Take-Off Distance Required (TODR)\n(JJA Data)")
#sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, img_name))


