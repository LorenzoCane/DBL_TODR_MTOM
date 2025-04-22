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
    - Description:
        This script performs a take-off distance analysis for various aircraft 
        masses using both manufacturer data and model estimates. It loads aircraft, 
        engine, and environmental configuration from a YAML config file, computes 
        the optimal lift coefficient (C_l) by minimizing RMSD between model and 
        reference data, and evaluates model performance through percentage error.

        Final results are exported as a Parquet file (with metadata), and a plot of 
        TODR vs. mass is generated as a PDF.

    - Main Components:
        - Config loading from config.yml
        - Aircraft/engine specification via OpenAP
        - TODR calculation via take_off() model
        - C_l optimization via cl_finder()
        - Error analysis and result plotting
        - Export to Apache Parquet with metadata

    - Dependencies:
        - numpy, pandas, matplotlib, pyarrow, yaml, openap, utils (local), take_off_func (local)

    - Outputs:
        - cl_TODR_data.parquet (model results + metadata)
        - TODR_mass.pdf (plot of TODR vs. aircraft mass)

    - Notes:
        - Replace config.yml with manual config values if needed
        - Ensure all necessary folders and data files exist before running (see README.md)


    ***************************************************************************
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openap import prop #aircraft and engine-related data
from openap.kinematic import WRAP #set of kinematic models
from openap.thrust import Thrust #thrust calc
from openap.drag import Drag

from pprint import pprint #“pretty-print” arbitrary Python data structures 

from utils import ComplexUnitConverter as conv
from utils import rmsd
from take_off_func import take_off, cl_finder

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
airport_alt = config['Airport']['airport_alt']

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']


print(f'Configuration successfully loaded from {config_file}')

'''
#***************************************************************************
#Manual config (keep it commented if not needed)
r_spec = 287.0 # (N*m) / (kg*K) 
#ISA conditions
temp_isa = conv.convert(15.0, 'celsius', 'kelvin') #K (15°C)
pres_isa = 101325 #Pa (1013.25 hPa)

pathway_incl = 0.0 #deg

asc = conv.convert(35.0, 'ft', 'm') # m
climb_ang = 7.7 # deg (see [Gratton et al. 2020])

safe_margin_coef = 1.15

os.makedirs('images', exist_ok=True)
os.makedirs('output_data', exist_ok=True)
img_path = './images/'
out_path = './output_data/'

aircraft_name = "A320"
engine_name = "V2500-A1"

#available_aircraft = prop.available_aircraft() #list of avaib aircraft if needed
#***************************************************************************
'''
#***************************************************************************
#Ensure dirs existance
os.makedirs(img_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
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
print(cd0)

#aircraft TO speeds
wrap = WRAP(ac=aircraft_name) #kinematic parameters
to_speed = wrap.takeoff_speed() # m/s Take-off speed. order: default (optimum), minimum, maximum
#print(to_speed)
opt_to_speed = to_speed['default'] #m/s
min_to_speed = to_speed['minimum'] #m/s
max_to_speed = to_speed['maximum'] #m/s
speed_val = np.sort([s for s in list(to_speed.values())[:3]])
print(speed_val)

#aircraft thrust
thr_a320 = Thrust(ac= aircraft_name, eng= engine_name)
T = np.array([thr_a320.takeoff(tas = conv.convert(i, 'ms', 'kts'), alt=0) for i in speed_val]) #N
print(T)

rho_isa = isa_pr / (r_spec * isa_temp) 

#***************************************************************************
# check TO distance calculation method (toy conditions)

aircraft_mass = np.array([61235., 63503.,65771., 68039., 70307., 72575., 74843., 77111., 79379.]) #kg
a_mass_err = np.ones(len(aircraft_mass))
to_manuf_value = [1233., 1344., 1455., 1579., 1689., 1798., 1946., 2134., 2362.,] # m
to_err = np.ones(len(to_manuf_value))

#dim check
if (len(aircraft_mass) != len(to_manuf_value)):
    raise ValueError (f"Dimension error: aircraft masses dim = {len(aircraft_mass)} != to manufacturer value = {len(to_manuf_value)}")
#print(len(aircraft_mass) == len(a_mass_err))


'''
#check     
test = take_off(aircraft_mass[2], T[1], rho_isa, 1.41, cd0, k, wing_area, airborne_dist, mu= mu, return_velocity=True)
print(test)
'''
#***************************************************************************
#Find best C_l values for min, opt and max take-off velocities
cl_values = []
cl_rsmd = []
print('-------------------------------------------------')
for i in range(0, len(T)):
    cl_val, err_cl = cl_finder(aircraft_mass, to_manuf_value, to_err, 
                               T[i], rho_isa, cd0, k, wing_area, airborne_dist, safe_margin_coef, v_takeoff=speed_val[i], mu = mu,
                               dv0=0.01, dv_decay='const', theta = 0.0, cl_min=1.0, cl_max=2.001, cl_step=0.001)

    cl_values.append(cl_val)
    cl_rsmd.append(err_cl)
print(cl_values)

#final results
cl_best = np.mean(cl_values)
err_cl_best = 0.5 * (np.max(cl_values) - np.min(cl_values))

print(f"C_l finding process results: C_l = {cl_best} +- {err_cl_best}")

#***************************************************************************
#Error analysis

#model prediction and  gt - model perc diff
model_to_dist = np.array([take_off(i, T[1], rho_isa, cl_best, cd0, k, wing_area, airborne_dist, safe_margin_coef, mu) for i in aircraft_mass])
perc_diff = (model_to_dist - to_manuf_value) / to_manuf_value * 100.0

print('-------------------------------------------------')
print('Perc. difference between Manufacturer and model values:' )
print(perc_diff)
print(f'Mean abs perc. difference: {np.mean(abs(perc_diff))} %')
print('-------------------------------------------------')
# Upper and lower errors from cl uncertainty
model_upper = np.array([
    take_off(m, T[1], rho_isa, np.min(cl_values), cd0, k, wing_area, airborne_dist, safe_margin_coef)
    for m in aircraft_mass
])
model_lower = np.array([
    take_off(m, T[1], rho_isa, np.max(cl_values), cd0, k, wing_area, airborne_dist, safe_margin_coef)
    for m in aircraft_mass
])

#Compute errors
model_err_upper = abs(model_upper - model_to_dist)
model_err_lower = abs(model_to_dist - model_lower)
#print(model_err_lower)
#print(model_err_upper)

#***************************************************************************
# Create a DataFrame with all the relevant data
df = pd.DataFrame({
    "mass_kg": aircraft_mass,
    "mass_tonnes": aircraft_mass / 1000.,
    "todr_manufacturer": to_manuf_value,
    "todr_model": model_to_dist,
    "todr_model_err_upper": model_err_upper,
    "todr_model_err_lower": model_err_lower,
    "todr_manufacturer_err": to_err,
})

# Add global values as metadata
metadata_dict = {
    "cl_best": cl_best,
    "cl_err": err_cl_best,
    "rho_isa": rho_isa,
    "T_used": T[1],
    "cd0": cd0,
    "k": k,
    "wing_area": wing_area,
    "airborne_dist": airborne_dist,
    "safe_margin_coef": safe_margin_coef,
    "mu": mu,
}

# Convert to Arrow Table (no index)
table = pa.Table.from_pandas(df, preserve_index=False)

# Add metadata (must be encoded as bytes)
encoded_meta = {str(k): str(v).encode("utf-8") for k, v in metadata_dict.items()}
existing_meta = table.schema.metadata or {}
merged_meta = {**existing_meta, **encoded_meta}
table = table.replace_schema_metadata(merged_meta)

# Save to parquet
parquet_path = os.path.join(output_path, "cl_TODR_data_vel_break.parquet")
pq.write_table(table, parquet_path)
print(f"Data with metadata written to {parquet_path}")

#***************************************************************************
#Plots
plt.figure()

# Model with asymmetric error bars
plt.errorbar(aircraft_mass/1000., model_to_dist, 
             yerr=[model_err_upper, model_err_lower],linestyle='',
             fmt='', capsize=4, label='Model data')

# Manufacturer data with symmetric error bars
plt.errorbar(aircraft_mass/1000., to_manuf_value, yerr=to_err, linestyle='--',
            fmt='o', capsize=4, label='Manufacturer data')

plt.xlabel('Aircraft mass [x 10^3 kg]')
plt.ylabel('TODR [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(img_path, "TODR_mass_vel_break.pdf"))


