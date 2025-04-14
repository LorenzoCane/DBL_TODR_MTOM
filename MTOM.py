'''

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

print(f'Configuration successfully loaded from {config_file}')
print('-------------------------------------------------')

#ensure path existance
os.makedirs(img_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
# Load the Parquet file
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
    print('-------------------------------------------------')
else:
    cl_best = 1.61
    print(f"No metadata found in the file.\n Default value C_l ={cl_best} will be used.")
    cl_best = 1.61
#cl_best = 1.61

#airborne dist
asc_m = conv.convert(asc_ft, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(climb_angle, 'deg', 'rad')) # m
#print(airborne_dist)

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
#MTOM calc w/ limited runway lenght
print('===========================================\nMTOM Calculation')
test_runway = np.linspace(1500.0, 2100.0, 13) #m
isa_rho = isa_pr / (r_spec * isa_temp)
print(f'Fixed TODR: {test_runway} m')

#Williams et al. code
start_time = time.monotonic()
mtom_cl = [mtom(lenght, aircraft_mass, T[1], isa_rho, cl_best, cd0, k , wing_area, airborne_dist,
            safe_margin_coef, mu, pathway_incl) for lenght in test_runway]
end_time = time.monotonic()
classic_time = timedelta(seconds = end_time - start_time).total_seconds()

#Binary search
start_time = time.monotonic()
mtom_bin = [mtom_binary(lenght, aircraft_mass, T[1], isa_rho, cl_best, cd0, k , wing_area, airborne_dist,
            safe_margin_coef, mu, pathway_incl, min_mass=50000, tol=1) for lenght in test_runway]
end_time = time.monotonic()
bin_time = timedelta(seconds = end_time - start_time).total_seconds()

print('MTOM calc. results')
for i in range(0,len(test_runway)):
    print(f'TODR[m] = {test_runway[i]}, MTOM_class[kg] = {mtom_cl[i]}, MTOM_bin[kg] = {mtom_bin[i]:.0f}')

print('Computational time:')
print(f'Classic method: {classic_time} s')
print(f'Binary method: {bin_time} s')