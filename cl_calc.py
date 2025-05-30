import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys
from config_loader import load_config
from todr import cl_finder, take_off
from aircraft_utils import get_aircraft_data, get_thrust, get_to_speed
from constants import *
from openap import prop #aircraft and engine-related data
sys.path.insert(0,'./utils')
from unit_converter import ComplexUnitConverter as conv

config = load_config()

cl_path = config['Dir']['cl_dir']
aircraft_name = config["Aircraft"]["aircr_name"]
engine_name = config["Aircraft"]["aircr_engine"]

aircraft_info = get_aircraft_data(aircraft_name, engine_name)

wing_area = aircraft_info["wing_area"]
cd0 = aircraft_info["cd0"]
k = aircraft_info["k"]
mu = aircraft_info["mu"]
mass_max = aircraft_info["mass_max"]

asc_m = conv.convert(ASC, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(CLIMB_ANGLE_DEG, 'deg', 'rad')) # m

rho_isa = ISA_PR / (R_SPEC * ISA_TEMP)

#Load aircraft masses and TO manuf results
manuf_file = cl_path + '/TODR_MTOM_manuf' + f"/{aircraft_name}_{engine_name}.txt"

if os.path.exists(manuf_file):
    manuf_data = np.loadtxt(manuf_file, comments="#", delimiter=",")
    aircraft_mass = manuf_data[:, 0]
    to_manuf_value = manuf_data[:, 1]
    to_err = np.ones(len(to_manuf_value))
else :
    raise ValueError (f"Manufacturer data for aircraft: {aircraft_name} - {engine_name} not found in {cl_path + 'TODR_MTOM_manuf'}")

#dim check
if (len(aircraft_mass) != len(to_manuf_value)):
    raise ValueError (f"Dimension error: aircraft masses dim = {len(aircraft_mass)} != to manufacturer value = {len(to_manuf_value)}")

#Find best C_l values for min, opt and max take-off velocities
T = get_thrust(aircraft_name, engine_name, 0)
cl_values = []
cl_rsmd = []

print('-------------------------------------------------')
for i in range(0, len(T)):
    cl_val, err_cl = cl_finder(aircraft_mass, to_manuf_value, to_err, 
                               T[i], rho_isa, cd0, k, wing_area, airborne_dist, MARGIN_COEFF,mu = mu,
                               dv0=0.01, dv_decay='const', theta = 0.0, cl_min=1.0, cl_max=2.0, cl_step=0.01)

    cl_values.append(cl_val)
    cl_rsmd.append(err_cl)
print(np.min(cl_values))

#final results
cl_best = np.mean(cl_values)
err_cl_best = 0.5 * (np.max(cl_values) - np.min(cl_values))

print(f"C_l finding process results: C_l = {cl_best} +- {err_cl_best}")

#model prediction and  gt - model perc diff
model_to_dist = np.array([take_off(i, T[1], rho_isa, cl_best, cd0, k, wing_area, airborne_dist, MARGIN_COEFF, mu=mu) for i in aircraft_mass])
perc_diff = (model_to_dist - to_manuf_value) / to_manuf_value * 100.0

'''
print('-------------------------------------------------')
print('Perc. difference between Manufacturer and model values:' )
print((perc_diff))
print(f'Mean abs perc. difference: {np.mean(abs(perc_diff)):.3f} %')
print('-------------------------------------------------')
'''
# Upper and lower errors from cl uncertainty
model_upper = np.array([
    take_off(m, T[1], rho_isa, cl_values[0], cd0, k, wing_area, airborne_dist, MARGIN_COEFF)
    for m in aircraft_mass
])
model_lower = np.array([
    take_off(m, T[1], rho_isa, cl_values[2], cd0, k, wing_area, airborne_dist, MARGIN_COEFF)
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
    "safe_margin_coef": MARGIN_COEFF,
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
parquet_path = os.path.join(cl_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")

#parquet_path = os.path.join(output_path, "cl_TODR_data_vel_break.parquet")

pq.write_table(table, parquet_path)
print(f"Data with metadata written to {parquet_path}")
