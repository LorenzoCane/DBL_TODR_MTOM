
import subprocess

from utils import ComplexUnitConverter as conv
from utils import rmsd , install_requirements
from atm_anal_func import process_scenario
from take_off_func import take_off, cl_finder
from grid_function import noise_grid, rotate_grid, project_to_latlon, plot_real_map

install_requirements(requirements_file='requirements.txt')
print('---------------------------------------------------------------------------')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
NORTH_DEG = 10.0 #deg
sep = '---------------------------------------------------------------------------'
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
os.makedirs(img_path, exist_ok=True)

# Load the Parquet file with C_L value
parquet_path = os.path.join(output_path, "cl_TODR_data.parquet")
table = pq.read_table(parquet_path)
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
    output_path=output_path,
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
    output_path=output_path,
    sel_months= climate_months
)
df_ssp370 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp370',
    var_pres_name='sp_ssp370',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code,
    scenario_name="SSP370",
    output_path=output_path,
    sel_months= climate_months
)
df_ssp585 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp585',
    var_pres_name='sp_ssp585',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code, 
    scenario_name="SSP585",
    output_path=output_path,
    sel_months= climate_months
)
print(sep)
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

'''
- (C_L calculation)?
- 
- Noise calculation
- Angle modification
- noise modification

Plots:
- Distributions (TODR and temp)
- MTOM restrictions
- Percentiles 
- Noise contours
- Noise modifications

'''