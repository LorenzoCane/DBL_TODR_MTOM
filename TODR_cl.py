'''
Lorenzo Cane    
Deep Blue srl

07/04/2025

'''

import intake
import xarray as xr
import numpy as np
import pandas as pd
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
r_spec = 287.0 # (N*m) / (kg*K) 
pathway_incl = 0.0 #deg

asc = conv.convert(35.0, 'ft', 'm') # m
climb_ang = 7.7 # deg (see [Gratton et al. 2020])

airborne_dist = asc / np.tan(conv.convert(climb_ang, 'deg', 'rad')) # m
#print(airborne_dist)

safe_margin_coef = 1.15

os.makedirs('images', exist_ok=True)
img_path = './images/'
#***************************************************************************
#config
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
print(cd0)

wrap = WRAP(ac=aircraft_name) #kinematic parameters

to_speed = wrap.takeoff_speed() # m/s Take-off speed. order: default (optimum), minimum, maximum
#print(to_speed)

opt_to_speed = to_speed['default'] #m/s
min_to_speed = to_speed['minimum'] #m/s
max_to_speed = to_speed['maximum'] #m/s
speed_val = np.sort([s for s in list(to_speed.values())[:3]])
#print(speed_val)
#----------------------------------------------------------------------------
#ISA conditions
temp_isa = conv.convert(15.0, 'celsius', 'kelvin') #K (15°C)
pres_isa = 101325 #Pa (1013.25 hPa)

rho_isa = pres_isa / (r_spec * temp_isa) 

#***************************************************************************
# check TO dostance calculation (toy conditions)

aircraft_mass = np.array([61235., 63503.,65771., 68039., 70307., 72575., 74843., 77111., 79379.]) #kg
a_mass_err = np.ones(len(aircraft_mass))
to_manuf_value = [1233., 1344., 1455., 1579., 1689., 1798., 1946., 2134., 2362.,] # m
to_err = np.ones(len(to_manuf_value))

#dim check
if (len(aircraft_mass) != len(to_manuf_value)):
    raise ValueError (f"Dimension error: aircraft masses dim = {len(aircraft_mass)} != to manufacturer value = {len(to_manuf_value)}")
#print(len(aircraft_mass) == len(a_mass_err))


thr_a320 = Thrust(ac= aircraft_name, eng= engine_name)
T = np.array([thr_a320.takeoff(tas = conv.convert(i, 'ms', 'kts'), alt=0) for i in speed_val]) #N
print(T)

#check     
test = take_off(aircraft_mass[2], T[1], rho_isa, 1.14, cd0, k, wing_area, airborne_dist, mu= mu, return_velocity=True)
print(test)

#***************************************************************************
#Find best C_l values for min, opt and max take-off velocities
cl_values = []
cl_rsmd = []
for thr in T:
    cl_val, err_cl = cl_finder(aircraft_mass, to_manuf_value, to_err, 
                               thr, rho_isa, cd0, k, wing_area, airborne_dist, safe_margin_coef, mu = mu,
                               theta = 0.0, cl_min=1.0, cl_max=2.0, cl_step=0.01)

    cl_values.append(cl_val)
    cl_rsmd.append(err_cl)


#final results
cl_best = np.mean(cl_values)
err_cl_best = 0.5 * (np.max(cl_values) - np.min(cl_values))

print(f"C_l finding process results: C_l = {cl_best} +- {err_cl_best}")

#***************************************************************************
#Error analysis

#model prediction and  gt - model perc diff
model_to_dist = np.array([take_off(i, T[1], rho_isa, cl_best, cd0, k, wing_area, airborne_dist, safe_margin_coef, mu) for i in aircraft_mass])
perc_diff = (to_manuf_value - model_to_dist) / to_manuf_value * 100.0
print(perc_diff)
# Upper and lower errors from cl uncertainty
model_upper = np.array([
    take_off(m, T[0], rho_isa, np.min(cl_values), cd0, k, wing_area, airborne_dist, safe_margin_coef)
    for m in aircraft_mass
])
model_lower = np.array([
    take_off(m, T[0], rho_isa, np.max(cl_values), cd0, k, wing_area, airborne_dist, safe_margin_coef)
    for m in aircraft_mass
])

# Asymmetric errors
model_err_upper = abs(model_upper - model_to_dist)
model_err_lower = abs(model_to_dist - model_lower)
#print(model_err_lower)
#print(model_err_upper)

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
plt.savefig(os.path.join(img_path, "TODR_mass.pdf"))
