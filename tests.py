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

sep = '--------------------------------------------------------------------'
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
#print(cd0)

#aircraft TO speeds
wrap = WRAP(ac=aircraft_name) #kinematic parameters
to_speed = wrap.takeoff_speed() # m/s Take-off speed. order: default (optimum), minimum, maximum
#print(to_speed)
opt_to_speed = to_speed['default'] #m/s
min_to_speed = to_speed['minimum'] #m/s
max_to_speed = to_speed['maximum'] #m/s
speed_val = np.sort([s for s in list(to_speed.values())[:3]])
#print(speed_val)

#aircraft thrust
thr_a320 = Thrust(ac= aircraft_name, eng= engine_name)
T = np.array([thr_a320.takeoff(tas = conv.convert(i, 'ms', 'kts'), alt=0) for i in speed_val]) #N
#print(T)

rho_isa = isa_pr / (r_spec * isa_temp) 

cl_best = 1.61

speed_arr_m = np.arange(40.0, 105.0, 5.0 )
speed_arr_m = np.sort(np.append(speed_arr_m, speed_val))
print(sep)
print('Velocities:')
print(speed_arr_m)
speed_arr_kts = [conv.convert(sp, 'ms', 'kts') for sp in speed_arr_m]
speed = 85.3 #m/s

drag = Drag(ac=aircraft_name)

# non- clean configuration
D_opap_nc = np.array([drag.nonclean(mass=aircraft_mass, tas=sp, flap_angle = 0.0, landing_gear= True, alt=0.0, vs=0.0) for sp in speed_arr_kts])
#clean config
D_opap_c = np.array([drag.clean(mass=aircraft_mass,  tas=sp, alt=0.0, vs=0.0) for sp in speed_arr_kts])

#cl = ((D_opap - 0.5 * rho_isa * speed * speed * cd0 * wing_area) * 2.0 / rho_isa / speed /speed/ k/ wing_area)**0.5

D_c = np.array([(0.5 * rho_isa * sp * sp * wing_area * (cd0 + k *cl_best * cl_best)) for sp in speed_arr_m])


#print(cl)

fig = plt.figure(figsize=(6,8))
ax = plt.subplot(211)
ax2 = ax.twiny()
ax.plot(speed_arr_m, D_c/1000, label= 'Parabolic drag model', linestyle = '--', color = 'orange')
ax2.plot(speed_arr_kts, D_opap_c/1000, label='Clean config (OPAP)', linestyle='--', color = 'blue')
ax2.plot(speed_arr_kts, D_opap_nc/1000, label='Non-clean config (OPAP)', linestyle='--', color='violet')
ax.set_xlabel('V [m/s]')
ax2.set_xlabel('V [kts]')
ax.set_ylabel('Drag Force [kN]')
ax.set_title(f'Drag vs. V - C_L = {cl_best}')
ax.set_ylim(10, 250)

for sp in speed_val:
    ax.axvline(x=sp, color = 'red', linestyle='-', linewidth=1.5)

ax.legend()
ax2.legend()
fig.tight_layout()
drag_img_path = os.path.join(img_path, f'Drag_V_cl_{str.replace(str(cl_best),'.','_')}.pdf')
fig.savefig(drag_img_path)
print(sep)
print(f'Image Drag vs. V saved in : {drag_img_path}')