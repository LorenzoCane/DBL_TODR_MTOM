import subprocess

from utils import ComplexUnitConverter as conv
from utils import rmsd , install_requirements
install_requirements(requirements_file='requirements.txt')
print('---------------------------------------------------------------------------')
from atm_anal_func import process_scenario
from take_off_func import take_off, mtom_binary, mtom
from grid_function import noise_grid, rotate_grid, project_to_latlon, plot_real_map

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import time
from datetime import timedelta
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning

import airportsdata #airport data
from openap import prop #aircraft and engine-related data
from openap.kinematic import WRAP #set of kinematic models
from openap.thrust import Thrust #thrust calc
from openap.drag import Drag
#***************************************************************************
#CONSTANTS
G = 9.81 #m/s^2
R_SPEC = 287.0528

CLIMB_ANGLE_DEG = 5.0 #deg
ENGINE_DB = 110 #dB
NORTH_DEG = 10.0 #deo
sep = '---------------------------------------------------------------------------'
CL_FINDER = False
#plots settings
n_bins_todr = 50
n_bins_atm = 100
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

pathway_incl = config['Constants']['pathway_incl']
asc_ft = config['Constants']['asc']
init_climb_angle = config['Constants']['climb_angle']
safe_margin_coef = config['Constants']['margin_coef']

isa_temp = config['ISA']['isa_temp']
isa_pr = config['ISA']['isa_pr']
isa_alt = config['ISA']['isa_alt']

aircraft_name = config['Aircraft']['aircr_name']
engine_name = config['Aircraft']['aircr_engine']
aircraft_mass = config['Aircraft']['aircr_full_m']

airport_code = config['Airport']['airport_code']
airport_length = config['Airport']['airport_lenght']

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

passenger_mass = config['Mass']['passenger_mass']
mass_restr_period = config['Mass']['period']

#airborne dist
asc_m = conv.convert(asc_ft, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(init_climb_angle, 'deg', 'rad')) # m

print(f'Configuration successfully loaded from {config_file}')
print(sep)
#***************************************************************************
#Ensure dir existance and create paths
os.makedirs(clim_data_dir, exist_ok=True)
os.makedirs(clean_data_dir, exist_ok=True) #output dir (clean data)
os.makedirs(output_path, exist_ok=True)

#Create a dir for each airport-aircraft combination
model_output_path = f'./AP_{airport_code}_AC_{aircraft_name}_{engine_name}'
model_plot_path = model_output_path + '/plots'
os.makedirs(model_output_path, exist_ok=True)
os.makedirs(model_plot_path, exist_ok=True)
#***************************************************************************
#Perform C_L evaluation if needed or request
cl_file_name = "cl_TODR_data.parquet"
cl_parquet_path = os.path.join(output_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")

if CL_FINDER or not os.path.exists(cl_parquet_path):
    print('==========================================================')
    print("Parquet file not found. Running CL evaluation script...")
    subprocess.run(["python", "TODR_cl.py"], stdout=subprocess.DEVNULL)  
    print('==========================================================')
   
# Load the Parquet file with C_L value
table = pq.read_table(cl_parquet_path)
# Extract and decode metadata
metadata = table.schema.metadata
if metadata:
    decoded_meta = {k.decode(): v.decode() for k, v in metadata.items()}
    cl_best = float(decoded_meta["cl_best"])
    cl_err = float(decoded_meta["cl_err"])
    print(f"C_l best: {cl_best}")
    print(f"C_l error: {cl_err}")
    print(sep)
else:
    cl_best = 1.61
    print(f"No metadata found in the file.\n Default value C_l ={cl_best} will be used.")
    cl_best = 1.61
#***************************************************************************
#Atm data 
file_path = os.path.join(clim_data_dir , f"cmip6_{climate_model}_{airport_code}.nc")
#Dataset
ds = xr.open_dataset(file_path)
sel_months = climate_months


# Process Historical data (1985-2014)
df_hist = process_scenario(
    ds,
    var_temp_name='mx2t24_historical',
    var_pres_name='sp_historical',
    time_range=("1985-01-01", "2014-12-31"),
    airport= airport_code,
    scenario_name="Historical",
    output_path=clean_data_dir,
    sel_months= climate_months
)

# Process SSP scenarios (2035-2064)
df_ssp126 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp126',
    var_pres_name='sp_ssp126',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code,
    scenario_name="SSP126",
    output_path=clean_data_dir,
    sel_months= climate_months
)
df_ssp370 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp370',
    var_pres_name='sp_ssp370',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code,
    scenario_name="SSP370",
    output_path=clean_data_dir,
    sel_months= climate_months
)
df_ssp585 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp585',
    var_pres_name='sp_ssp585',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code, 
    scenario_name="SSP585",
    output_path=clean_data_dir,
    sel_months= climate_months
)

#***************************************************************************
#TODR & MTOM calculation
#airport data from airports library
airports = airportsdata.load()  # key is the ICAO identifier (the default) 

selected_airport = airports[airport_code] #LIPE = Bologna Borgo Panigale
airport_name = selected_airport['name']
airport_lat = selected_airport['lat']
airport_long = selected_airport['lon']
airport_alt_ft = selected_airport['elevation'] #ft MSL elevation of the highest point of the landing area, in feet (warning: it is often wrong);

#airport runway lenght from personal database
df_airport = pd.read_csv(os.path.join(clim_data_dir, "airport_runways.csv"))
airport_length_m = df_airport.loc[df_airport['ICAO'] == airport_code, "Runway_Length_m"].values[0] #m
if airport_length_m < 0 : 
    raise ValueError(f'ICAO code {airport_code} not found.')

print(sep)
print(f'Airport: {airport_name} ({airport_code}) data loaded')

#aircraft module from OpenAP
engine = prop.engine(engine_name) #engine dict
aircraft = prop.aircraft(aircraft_name) #aircraft dict
wing_area = aircraft['wing']['area']
cd0 = aircraft['drag']['cd0']
k = aircraft['drag']['k']
mu = aircraft['drag']['gears']
aircraft_full_mass = aircraft['limits']['MTOW']

#aircraft TO speed
wrap = WRAP(ac=aircraft_name) #kinematic parameters
to_speed = wrap.takeoff_speed() # m/s Take-off speed. order: default (optimum), minimum, maximum
#print(to_speed)
opt_to_speed = to_speed['default'] #m/s
min_to_speed = to_speed['minimum'] #m/s
max_to_speed = to_speed['maximum'] #m/s
speed_val = np.sort([s for s in list(to_speed.values())[:3]])

print(sep)
print(f'Aircraft: {aircraft_name}-{engine_name} data loaded')
#Thrust
thr_a320 = Thrust(ac= aircraft_name, eng= engine_name)
T = np.array([thr_a320.takeoff(tas = i, alt=airport_alt_ft) for i in speed_val]) #N

#---------------------------------------------------------------------------------------------
#Re-read csv files
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
    print(f'Creating DataFrame for scenario: {scenario}')
    # Compute air density: rho = Pressure / (R * Temperature)
    df["rho"] = df["sp"] / (R_SPEC * df["mx2t24"])
    print("Air densities evaluated")
    # Compute TODR for each row using the same constant parameters
    df["TODR"] = df.apply(lambda row: take_off(aircraft_mass, T[1], row["rho"], cl_best, cd0, k, wing_area, 
                                               airborne_dist, safe_margin_coef, mu, pathway_incl), axis=1)
    print("TODR evaluated")
    # Compute MTOM for each row
    df["MTOM"] = df.apply(lambda row: mtom(airport_length, aircraft_mass, T[1], row["rho"], cl_best, cd0, k,
                                                  wing_area, airborne_dist, safe_margin_coef, mu, pathway_incl), axis = 1)
    print("MTOM evaluated")
    # Compute mass reduction in kg and n. of passenger
    df["mass_restr_kg"] = df["MTOM"] - aircraft_mass #kg Negative numbers
    
    df["mass_restr_pass"] = df['mass_restr_kg'] // passenger_mass #being neg counts one more "cancelled passanger" (conservative way)
    print('Mass restriction evaluated')
    # Add a column for the scenario label
    df["Scenario"] = scenario
    print(sep)
    # Remove rows where TODR, temperature, or pressure are NaN
    #df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    all_data.append(df)

# Concatenate all the scenario DataFrames
df_all = pd.concat(all_data, ignore_index=True)
print("\n=== TODR Summary by Scenario ===")
print(df_all.groupby("Scenario")["TODR"].describe().round(2))
print(sep)

#Save data for future plots and anal
performance_parquet_name = f"{airport_code}_{aircraft_name}_{engine_name}_TODR_MTOM.parquet"
performance_parquet_path = os.path.join(model_output_path, performance_parquet_name)

df_all.to_parquet(performance_parquet_path)
print(sep)
print(f'All processed data saved in {performance_parquet_path}')

#***************************************************************************
#ROC PREDICTION
#***************************************************************************
#NOISE IMPACT
#***************************************************************************
#PLOTS
#C_l finder plots
df = pd.read_parquet(cl_parquet_path)
img_out = os.path.join(model_plot_path, f'TODR_mass_{aircraft_name}_{engine_name}.pdf')

plt.figure()

# Model data with asymmetric errors
plt.errorbar(df["mass_tonnes"], df["todr_model"],
             yerr=[df["todr_model_err_upper"], df["todr_model_err_lower"]],
             linestyle='', fmt='', capsize=4, label='Model data', color='tab:blue')

# Manufacturer data with symmetric error
plt.errorbar(df["mass_tonnes"], df["todr_manufacturer"],
             yerr=df["todr_manufacturer_err"], linestyle='--',
            fmt='o', capsize=4, label='Manufacturer data', color='tab:orange')

plt.xlabel('Aircraft mass [10^3 kg]')
plt.ylabel('TODR [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(img_out)
# plt.show()  

#*********************************************************************
#TODR and MTOM w/ scenario plot

df_all = pd.read_parquet(performance_parquet_path)
img_name = f'{airport_code}_{aircraft_name}_{engine_name}_TODR_boxplot.pdf'
img_out = os.path.join(model_plot_path, img_name)

#Create the Boxplot
sns.set_style("whitegrid")
#print(img_out)
plt.figure(figsize=(8, 6))

# Define custom order and colors for scenarios if desired
order = ["Historical", "SSP126", "SSP370", "SSP585"]
palette = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]

sns.boxplot(x="Scenario", y="TODR", data=df_all, whis =1.5, order=order, palette=palette, showfliers=True)
plt.xlabel("Scenario")
plt.ylabel("TODR [m]")
plt.title(f"TODR (JJA) {aircraft_name} - {engine_name} - {airport_code} - {id}")
#sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(img_path, img_name))

#---------------------------------------------------------------------------
#TODR distr. hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_TODR_hist.pdf"

# Get the unique scenarios
scenarios = df_all['Scenario'].unique()
n_scenarios = len(scenarios)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['TODR'], bins=n_bins_todr, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("TODR [m]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(model_plot_path, hist_name))

    
#temp hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_temp_hist.pdf"

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['mx2t24'], bins=n_bins_atm, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("T [K]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(model_plot_path, hist_name))


#sur pres hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_sp_hist.pdf"
    
# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['sp'], bins=n_bins_atm, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("sp [Pa]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(model_plot_path, hist_name))

#rho pres hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_rho_hist.pdf"

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['rho'], bins=n_bins_atm, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("air density [kg m^-3]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.tight_layout()
plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.savefig(os.path.join(model_plot_path, hist_name))

print(f'Images saved in {model_plot_path}')

#---------------------------------------------------------------------------
#MTOM and mass restriction 
df_all = pd.read_parquet(performance_parquet_path)
#Compute period means for plot
period_m_restr = (df_all.groupby(['Scenario', mass_restr_period])[['mass_restr_kg', 'mass_restr_pass']]
                  .mean().reset_index()
                  )
mass_restr_scenarios = ["Historical", "SSP126", "SSP370", "SSP585"] #include 'Historical' if needed

fig, axes = plt.subplots(nrows=len(mass_restr_scenarios), ncols=1)

for ax, scenario in zip(axes, mass_restr_scenarios):
    df_s = period_m_restr[period_m_restr['Scenario'] == scenario]

    # Left axis: mass_restr_kg
    ln1 = ax.plot(df_s['Year'], df_s['mass_restr_kg'], linestyle='--',
                    label='Mass Restriction [kg]')
    #ax.grid(True)

    # Right axis: mass_restr_pass
    ax2 = ax.twinx()
    ln2 = ax2.plot(df_s['Year'], df_s['mass_restr_pass'], linestyle='-',
        label='Passenger Restriction [#]')
    
    ax2.set_ylim(-126,-101)
    ax.set_ylim(-126 * passenger_mass, -101 * passenger_mass)
    ax.set_title(f"{scenario} Scenario")

# Common labels
fig.text(0.5, 0.02, 'Year', ha='center')
fig.text(0.02, 0.5, 'Mass restriction [kg]', va='center', rotation='vertical')
fig.text(0.98, 0.5, 'Passenger Restriction [#]', va='center', rotation='vertical')
plt.tight_layout()

mtom_img_name = f'{airport_code}_{aircraft_name}_{engine_name}_MTOM_restr_pass_m{passenger_mass}.pdf'
plt.savefig(os.path.join(model_plot_path, mtom_img_name))

#-----------------------------------------------------------------------------------------
#Hist vs. scenarios comparison
# Compute the Historical‐scenario overall means
hist = period_m_restr[period_m_restr['Scenario'] == 'Historical']
hist_mean_kg   = hist['mass_restr_kg'].mean()
hist_mean_pass = hist['mass_restr_pass'].mean()

# Build a pivot of the SSP scenarios
ssp_scenarios = ['SSP126','SSP370','SSP585']
pivot = period_m_restr.pivot(index='Year', columns='Scenario', values=['mass_restr_kg','mass_restr_pass'])

# Subtract the historical‐mean constant
diff_kg   = pivot['mass_restr_kg'][ssp_scenarios]   - hist_mean_kg
diff_pass = diff_kg / passenger_mass

# Plot
fig, axes = plt.subplots(
    nrows=len(ssp_scenarios), ncols=1)

for ax, scen in zip(axes, ssp_scenarios):
    # Left axis: mass difference
    ln1 = ax.plot(diff_kg.index, diff_kg[scen], linestyle='-',
        label='Δ Mass Restriction [kg]')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    #ax.grid(True)

    # Right axis: passenger difference
    ax2 = ax.twinx()
    ln2 = ax2.plot(diff_pass.index, diff_pass[scen], linestyle='-',
        label='Δ Passenger Restriction [#]')
    ax2.set_ylim(-20, -3)

    
    # "Sync" axes
    ax.set_ylim(-20 *passenger_mass, -3*passenger_mass)
    
    ax.set_title(f"{scen} - additional mass restriction")

# Common  labels
fig.text(0.5, 0.02, 'Year', ha='center')
fig.text(0.02, 0.5, 'Additional Mass restriction [kg]', va='center', rotation='vertical')
fig.text(0.98, 0.5, 'Additional Passenger Restriction [#]', va='center', rotation='vertical')
plt.tight_layout()

mtom_img_name = f'{airport_code}_{aircraft_name}_{engine_name}_ADD_MTOM_restr_pass_m{passenger_mass}.pdf'
plt.savefig(os.path.join(model_plot_path, mtom_img_name))
#***************************************************************************

'''
- Noise calculation
- Angle modification
- noise modification

Plots:
- Percentiles 
- Noise contours
- Noise modifications
'''