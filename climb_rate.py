#CLIMB RATE & ANGLE OF CLIMB

#import libraries and internal files

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import yaml
import os, sys
import time
from datetime import timedelta

import airportsdata #airport data
from openap import prop #aircraft and engine-related data
from openap.kinematic import WRAP #set of kinematic models
from openap.thrust import Thrust #thrust calc
from openap.drag import Drag

from utils import ComplexUnitConverter as conv
from take_off_func import take_off
from grid_function import noise_grid, rotate_grid, project_to_latlon, plot_real_map
#***************************************************************************
#Constants

G = 9.81 #m/s^2
R_SPEC = 287.0528

pathway_incl = 0.0
CLIMB_ANGLE_DEG = 5.0 #deg
ENGINE_DB = 110 #dB
NORTH_DEG = 10.0 #deg
sep = '---------------------------------------------------------------------------'
#---------------------------------------------------------------------------
#import from configuration file config.yml
config_file = 'config.yml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

img_path = config['Dir']['img_dir']
output_path =  config['Dir']['output_dir']
clim_data_dir = config['Dir']['clim_data_dir']
clean_data_dir = config['Dir']['clean_data_dir']

asc_ft = config['Constants']['asc']
init_climb_angle = config['Constants']['climb_angle']
safe_margin_coef = config['Constants']['margin_coef']

airport_code = config['Airport']['airport_code']

aircraft_name = config['Aircraft']['aircr_name']
engine_name = config['Aircraft']['aircr_engine']

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

isa_temp = config['ISA']['isa_temp']
isa_pr = config['ISA']['isa_pr']
isa_alt = config['ISA']['isa_alt']

dv0 = config['Speed']['v0']
dv_decay = config['Speed']['decay']

print(f'Configuration successfully loaded from {config_file}')
#--------------------------------------------------------------------------
#Calc usefull quantities
#airborne dist
asc_m = conv.convert(asc_ft, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(init_climb_angle, 'deg', 'rad')) # m
#--------------------------------------------------------------------------
#Create dirs (if necessary)
os.makedirs(img_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
#---------------------------------------------------------------------------
#aircraft module from OpenAP
engine = prop.engine(engine_name) #engine dict
aircraft = prop.aircraft(aircraft_name) #aircraft dict
wing_area = aircraft['wing']['area']
cd0 = aircraft['drag']['cd0']
k = aircraft['drag']['k']
mu = aircraft['drag']['gears']
aircraft_full_mass = aircraft['limits']['MTOW']

print(sep)
print(f'Aircraft: {aircraft_name}-{engine_name} data loaded')
#---------------------------------------------------------------------------
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
#--------------------------------------------------------------------------
#Load C_L extimation data
parquet_path = os.path.join(output_path, "cl_TODR_data.parquet")
table = pq.read_table(parquet_path)

# Extract and decode metadata
metadata = table.schema.metadata
print(sep)
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

#---------------------------------------------------------------------------
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
T = np.array([thr_a320.takeoff(tas = i, alt=airport_alt_ft) for i in speed_val]) #N

#***************************************************************************
#Access climate data & create dataframe

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
    df["TODR_raw"] = df.apply(lambda row: take_off(
    aircraft_full_mass, T[1], row["rho"], cl_best, cd0, k, wing_area,
    airborne_dist, safe_margin_coef, mu, pathway_incl, dv0=dv0, dv_decay=dv_decay), axis=1)
    #stores the final TODR value, capped at the runway lenght
    df["TODR"] = df["TODR_raw"].apply(lambda todr: todr if todr < airport_length_m else airport_length_m)
    print("TODR evaluated")
    # Add a column for the scenario label
    df["Scenario"] = scenario
    # Remove rows where TODR, temperature, or pressure are NaN
    #df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    all_data.append(df)

    if scenario == "Historical":
        all_lpmax_grids = []
        grid_scale = 10
        grid_res = 200
        extra_frac = 0.25
        for i, row in df.iterrows():
            todr = row["TODR"]
            X, Y, Lp_ts, Lp_max, *_ = noise_grid(
                runway_length=airport_length_m,
                TODR=todr,
                climb_angle_deg=CLIMB_ANGLE_DEG,
                sound_level=ENGINE_DB,
                grid_scale=10,
                grid_points=200,
                extra_frac=0.25,
                npoints=300
            )
            all_lpmax_grids.append(Lp_max)

        final_max_grid = np.max(np.array(all_lpmax_grids), axis=0)
        np.save(output_path + "/final_max_noise_grid.npy", final_max_grid)

        final_mean_grid = np.mean(np.array(all_lpmax_grids), axis=0)
        np.save(output_path + "/final_mean_noise_grid.npy", final_mean_grid)

        # Rotate, project and plot
        X_rot, Y_rot = rotate_grid(X, Y, NORTH_DEG)
        lat_grid, lon_grid = project_to_latlon(X_rot, Y_rot, airport_lat, airport_long)
        output_name = f'{airport_code}_noise_contour_HISTORICAL.pdf'
        plot_real_map(lat_grid, lon_grid, final_max_grid, airport_lat, airport_long,
                      output_name, img_path)


# Concatenate all the scenario DataFrames
df_all = pd.concat(all_data, ignore_index=True)
#print("\n=== TODR Summary by Scenario ===")
#print(df_all.groupby("Scenario")["TODR"].describe().round(2))

#Save data for future plots and anal
save_path = os.path.join(output_path, f"{airport_code}_climb_angle_condition.parquet")

df_all.to_parquet(save_path)
print(sep)
print(f'All processed data saved in {save_path}')

'''
#***************************************************************************
#Calculate noise grid
all_lpmax_grids = []
grid_scale = 10
grid_res = 200
extra_frac = 0.25


for i, row in df.iterrows():
    todr = row["TODR"]

    # Compute noise grid
    X, Y, Lp_ts, Lp_max, *_ = noise_grid(
        runway_length=airport_length_m,
        TODR=todr,
        climb_angle_deg=CLIMB_ANGLE_DEG,
        sound_level=ENGINE_DB,
        grid_scale=grid_scale,
        grid_points=grid_res,
        extra_frac=extra_frac,
        npoints=300
    )

    # Store the Lp_max grid
    all_lpmax_grids.append(Lp_max)
    
    # Optionally save each one
    #np.save(output_path / f"row_{i}_Lpmax.npy", Lp_max)
    

#Compute final grid(ss)
final_max_grid = np.max(np.array(all_lpmax_grids), axis=0)
np.save(output_path + "/final_max_noise_grid.npy", final_max_grid)

# Or mean noise
final_mean_grid = np.mean(np.array(all_lpmax_grids), axis=0)
np.save(output_path + "/final_mean_noise_grid.npy", final_mean_grid)

#***************************************************************************
X_rot, Y_rot = rotate_grid(X, Y, NORTH_DEG)
lat_grid, lon_grid = project_to_latlon(X_rot, Y_rot, airport_lat, airport_long)

output_name = f'{airport_code}_noise_contour.pdf'
plot_db_contours(lat_grid, lon_grid, final_max_grid, output_name, output_path)
'''

#Code scheme
'''

noise grid (should be autom)
-calc noise map for each TODR (max in each point)
-take mean value for each grid point (over time)
-plot contour
-add to a map



rotate and add to map (how)

'''
#***************************************************************************
#***************************************************************************