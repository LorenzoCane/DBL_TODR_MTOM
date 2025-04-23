'''    
    ***************************************************************************
    Project : AEROPLANE
    Module  : TODR, MTOM and CC
    Author  : Lorenzo Cane
    Company : Deep Blue srl
    Created : 11/04/2025
    Updated : 14/04/2025
    Version : v1.0.0
    -------------------------------------------------------------------------------
    - Descrition:
        This script performs Take-Off Distance Required (TODR) and Maximum Take-Off Mass (MTOM)
        calculations for a selected aircraft in a selected airport. It analyzes the change of TODR 
        in the future caused by different climate scenarios.

        Final results are exported as a Parquet file, and boxplots of TODR vs scenarios and
        histograms of TODR and atmospoheric conditions are generated as PDF files.

    - Main Components:
        - Config loading from config.yml
        - Aircraft/engine specification via OpenAP
        - Computes key performance metrics using custom functions such as take_off.
        - Reads and processes input climate and scenario data from CSV files.
        - Applies climate corrections via density and environmental parameters.
        - Saves processed data (including metadata) to Apache Parquet format.
        - Generates various plots (boxplots and histograms) to visualize TODR distributions under different climate scenarios.
    
    - Dependencies: 
        - numpy, pandas, seaborn, matplotlib, pyarrow, yaml, warnings, utils (local), take_off_func (local)

    - Notes:
        - The module is configured to suppress Seaborn warnings.
        - Replace config.yml with manual config values if needed
        - Ensure all necessary folders and data files exist before running (see README.md)
    ***************************************************************************
'''



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import multiprocessing as mp
import time
from datetime import timedelta
import pandas as pd
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning

from openap import prop #aircraft and engine-related data
from openap.kinematic import WRAP #set of kinematic models
from openap.thrust import Thrust #thrust calc

from pprint import pprint #“pretty-print” arbitrary Python data structures 

from utils import ComplexUnitConverter as conv
from take_off_func import take_off, mtom, mtom_binary

#***************************************************************************
#import from configuration file config.yml
config_file = 'config.yml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)


# Accessing different sections
img_path = config['Dir']['img_dir']
output_path = config['Dir']['output_dir']
clim_data_dir = config['Dir']['clim_data_dir']
clean_data_dir = config['Dir']['clean_data_dir']

r_spec = config['Constants']['r_spec']
pathway_incl = config['Constants']['pathway_incl']
asc_ft = config['Constants']['asc']
climb_angle = config['Constants']['climb_angle']
safe_margin_coef = config['Constants']['margin_coef']

isa_temp = config['ISA']['isa_temp']
isa_pr = config['ISA']['isa_pr']
isa_alt = config['ISA']['isa_alt']

aircraft_name = config['Aircraft']['aircr_name']
engine_name = config['Aircraft']['aircr_engine']
aircraft_mass = config['Aircraft']['aircr_full_m']

airport_code = config['Airport']['airport_code']
airport_length = config['Airport']['airport_lenght']
airport_alt_m = config['Airport']['airport_alt']
airport_alt_ft = conv.convert(airport_alt_m, 'm', 'ft')

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

dv0 = config['Speed']['v0']
dv_decay = config['Speed']['decay']

print(f'Configuration successfully loaded from {config_file}')
print('-------------------------------------------------')

#ensure path existance
os.makedirs(img_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
# Load the Parquet file (choose the right one)
parquet_path = os.path.join(output_path, "cl_TODR_data.parquet")
#parquet_path = os.path.join(output_path, "cl_TODR_data_vel_break.parquet") #vel break
table = pq.read_table(parquet_path)
# Extract and decode metadata
metadata = table.schema.metadata
if metadata:
    decoded_meta = {k.decode(): v.decode() for k, v in metadata.items()}
    cl_best = float(decoded_meta["cl_best"])
    cl_err = float(decoded_meta["cl_err"])
    print(f"C_l best: {cl_best}")
    print(f"C_l error: {cl_err}")
    print('-------------------------------------------------')
else:
    cl_best = 1.61
    print(f"No metadata found in the file.\n Default value C_l ={cl_best} will be used.")
    cl_best = 1.61
#cl_best = 1.41

#airborne dist
asc_m = conv.convert(asc_ft, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(climb_angle, 'deg', 'rad')) # m
#print(airborne_dist)

'''
#***************************************************************************
#Manual config (keep it commented if not needed)
airport_code = 'EBBR' #Bruxells
full_mass = 78000.0 #kg
brussels_lenght = 3638.0 #m
airport_alt = conv.convert(56.0, 'm', 'ft') #m


#Define path
os.makedirs('images', exist_ok=True)
os.makedirs('output_data', exist_ok=True)
img_dir = './images/'
input_dir = './data/clean'
img_path = './images/'
out_dir = './output_data/'


r_spec = 287.0 # (N*m) / (kg*K) 
pathway_incl = 0.0 #deg

asc = conv.convert(35.0, 'ft', 'm') # m
climb_ang = 7.7 # deg (see [Gratton et al. 2020])

airborne_dist = asc / np.tan(conv.convert(climb_ang, 'deg', 'rad')) # m
#print(airborne_dist)

safe_margin_coef = 1.15
aircraft_name = "A320"
engine_name = "V2500-A1"

#available_aircraft = prop.available_aircraft() #list of avaib aircraft if needed
'''

#***************************************************************************
#aircraft specif.
aircraft = prop.aircraft(aircraft_name) #airbus A320
#pprint(aircraft)
engine = prop.engine(engine_name) #V2500-A1 turbofan engines
wing_area = aircraft['wing']['area'] #wing area
cd0 = aircraft['drag']['cd0']
k = aircraft['drag']['k']
mu = aircraft['drag']['gears']
#print(k)

#aircraft TO speed
wrap = WRAP(ac=aircraft_name) #kinematic parameters
to_speed = wrap.takeoff_speed() # m/s Take-off speed. order: default (optimum), minimum, maximum
#print(to_speed)
opt_to_speed = to_speed['default'] #m/s
min_to_speed = to_speed['minimum'] #m/s
max_to_speed = to_speed['maximum'] #m/s
speed_val = np.sort([s for s in list(to_speed.values())[:3]])
#print(speed_val)

#Thrust
thr_a320 = Thrust(ac= aircraft_name, eng= engine_name)
T = np.array([thr_a320.takeoff(tas = conv.convert(i, 'ms', 'kts'), alt=airport_alt_ft) for i in speed_val]) #N

#***************************************************************************
#Acces .csv file
file_dict = {
    "Historical": airport_code + "_Historical_JJA.csv",
    "SSP126": airport_code + "_SSP126_JJA.csv",
    "SSP370": airport_code + "_SSP370_JJA.csv",
    "SSP585": airport_code + "_SSP585_JJA.csv"
}


all_data = []  # list to hold each scenario's processed DataFrame
for scenario, filename in file_dict.items():
    # Read the CSV; each file should have at least columns: "mx2t24" (temperature, [K]) and "sp" (pressure, [Pa])
    df = pd.read_csv(os.path.join(clean_data_dir, filename))
    
    # Compute air density: rho = Pressure / (R * Temperature)
    df["rho"] = df["sp"] / (r_spec * df["mx2t24"])
    
    # Compute TODR for each row using the same constant parameters
    df["TODR"] = df.apply(lambda row: take_off(aircraft_mass, T[1], row["rho"], cl_best, cd0, k, wing_area, 
                                               airborne_dist, safe_margin_coef, mu, pathway_incl, dv0=dv0, dv_decay=dv_decay), axis=1)
    
    # Add a column for the scenario label
    df["Scenario"] = scenario
    
    # Remove rows where TODR, temperature, or pressure are NaN
    #df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    all_data.append(df)

# Concatenate all the scenario DataFrames
df_all = pd.concat(all_data, ignore_index=True)
print("\n=== TODR Summary by Scenario ===")
print(df_all.groupby("Scenario")["TODR"].describe().round(2))
print('-------------------------------------------------')


#Save data for future plots and anal
save_path = os.path.join(output_path, f"{airport_code}_TODR_NOQDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}.parquet")

#save_path = os.path.join(output_path, f"{airport_code}_TODR_NOQDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_vel_break.parquet")

df_all.to_parquet(save_path)
print(f'All processed data saved in {save_path}')
print('-------------------------------------------------')

'''
#***************************************************************************
#Create the Boxplot and hist
img_name = f'{airport_code}_NOQDM.pdf'

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
plt.savefig(os.path.join(img_path, img_name))

#---------------------------------------------------------------------------
n_bins = 50
#TODR distr. hist
hist_name = f"{airport_code}_TODR_NOQDM_hist.pdf"
# Get the unique scenarios
scenarios = df_all['Scenario'].unique()
n_scenarios = len(scenarios)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['TODR'], bins=n_bins, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("TODR")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(img_path, hist_name))

#temp hist
hist_name = f"{airport_code}_temp_NOQDM_hist.pdf"
# Get the unique scenarios
scenarios = df_all['Scenario'].unique()
n_scenarios = len(scenarios)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['mx2t24'], bins=n_bins, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("T [K]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(img_path, hist_name))


#sur pres hist
hist_name = f"{airport_code}_sp_NOQDM_hist.pdf"
# Get the unique scenarios
scenarios = df_all['Scenario'].unique()
n_scenarios = len(scenarios)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['sp'], bins=n_bins, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("sp [Pa]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(img_path, hist_name))

#rho pres hist
hist_name = f"{airport_code}_rho_NOQDM_hist.pdf"
# Get the unique scenarios
scenarios = df_all['Scenario'].unique()
n_scenarios = len(scenarios)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['rho'], bins=n_bins, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("air density [kg m^-3]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(img_path, hist_name))
'''

'''
#***************************************************************************
#with QDM
file_dict = {
    "Historical": airport_code + "_Historical_QDM_JJA.csv",
    "SSP126": airport_code + "_SSP126_QDM_JJA.csv",
    "SSP370": airport_code + "_SSP370_QDM_JJA.csv",
    "SSP585": airport_code + "_SSP585_QDM_JJA.csv"
}

img_name = f'{airport_code}_QDM.pdf'

all_data = []  # list to hold each scenario's processed DataFrame
for scenario, filename in file_dict.items():
    # Read the CSV; each file should have at least columns: "mx2t24" (temperature, [K]) and "sp" (pressure, [Pa])
    df = pd.read_csv(os.path.join(clean_data_dir, filename))
    
    # Compute air density: rho = Pressure / (R * Temperature)
    df["rho"] = df["sp"] / (r_spec * df["mx2t24"])
    
    # Compute TODR for each row using the same constant parameters
    df["TODR"] = df.apply(lambda row: take_off(aircraft_mass, T[1], row["rho"], cl_best, cd0, k, wing_area, airborne_dist, 
                                               safe_margin_coef, mu, pathway_incl, dv0=dv0, dv_decay=dv_decay), axis=1)
    
    # Add a column for the scenario label
    df["Scenario"] = scenario
    
    # Remove rows where TODR, temperature, or pressure are NaN
    df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    all_data.append(df)

# Concatenate all the scenario DataFrames
df_all = pd.concat(all_data, ignore_index=True)

#Save for future
#df_all.to_parquet(os.path.join(output_path, f"{airport_code}_TODR_QDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}.parquet"))

df_all.to_parquet(os.path.join(output_path, f"{airport_code}_TODR_QDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_vel_break.parquet"))
'''
'''

#Create the Boxplot
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
plt.savefig(os.path.join(img_path, img_name))

print(f'Images saved in {img_path}')
'''

