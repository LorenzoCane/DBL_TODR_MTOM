import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import time
from datetime import timedelta
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
from take_off_func import take_off, cl_finder, mtom, mtom_binary, take_off_modified, trajectory_s2n, grid_def

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
print(f'Speeds: {speed_val}')

#aircraft thrust
thr_a320 = Thrust(ac= aircraft_name, eng= engine_name)
T = np.array([thr_a320.takeoff(tas = i, alt=0) for i in speed_val]) #N
#print(T)

rho_isa = isa_pr / (r_spec * isa_temp) 

cl_best = 1.61

speed_arr_m = np.arange(30.0, 205.0, 2.0 )
speed_arr_m = np.sort(np.append(speed_arr_m, speed_val))
print(sep)
print('Velocities:')
print(speed_arr_m)
speed_arr_kts = np.array([conv.convert(sp, 'ms', 'kts') for sp in speed_arr_m])
#speed = 85.3 #m/s
print(speed_arr_kts)

#*************************************************************************************************
#Drag test
drag = Drag(ac=aircraft_name)

# non- clean configuration
D_opap_nc = np.array([drag.nonclean(mass=aircraft_mass, tas=sp, flap_angle = 0.0, landing_gear= True, alt=0.0, vs=0.0) for sp in speed_arr_kts])
#clean config
D_opap_c = np.array([drag.clean(mass=aircraft_mass,  tas=sp, alt=0.0, vs=0.0) for sp in speed_arr_kts])

#cl = ((D_opap - 0.5 * rho_isa * speed * speed * cd0 * wing_area) * 2.0 / rho_isa / speed /speed/ k/ wing_area)**0.5

D_c = np.array([(0.5 * rho_isa * sp * sp * wing_area * (cd0 + k *cl_best * cl_best)) for sp in speed_arr_m])


#print(cl)

fig = plt.figure()
ax = plt.subplot()
ax2 = ax.twiny()
ax.plot(speed_arr_m, D_c/1000, label= 'Parabolic drag model', linestyle = '--', color = 'orange')
ax2.plot(speed_arr_kts, D_opap_c/1000, label='Clean config (OPAP)', linestyle='--', color = 'blue')
ax2.plot(speed_arr_kts, D_opap_nc/1000, label='Non-clean config (OPAP)', linestyle='--', color='violet')
ax.set_xlabel('V [m/s]')
ax2.set_xlabel('V [kts]')
ax.set_ylabel('Drag Force [kN]')
ax.set_title(f'Drag vs. speed - C_L = {cl_best}')
ax.set_ylim(10, 250)

for sp in speed_val:
    ax.axvline(x=sp, color = 'red', linestyle='-', linewidth=1.5)

ax.legend()
ax2.legend()
fig.tight_layout()
drag_img_path = os.path.join(img_path, f'Drag_V_cl_{str.replace(str(cl_best),'.','_')}.pdf')
fig.savefig(drag_img_path)
print(sep)
print(f'Image Drag vs. Speed saved in : {drag_img_path}')

#*************************************************************************************************
#Thrust Tests
thr = np.array([thr_a320.takeoff(tas = sp, alt=0) for sp in speed_arr_kts]) #N

fig2 = plt.figure()
ax3 = plt.subplot()
ax4 = ax3.twiny()
ax3.plot(speed_arr_m, thr/1000.0, label= 'Thrust (OpenAP)', linestyle = '--', color = 'blue')
ax4.plot(speed_arr_kts, thr/1000.0, linestyle='--', color = 'blue')
ax3.set_xlabel('V [m/s]')
ax4.set_xlabel('V [kts]')
ax3.set_ylabel('Thrust [kN]')
ax3.set_title(f'Thrust vs. speed - Altitude = 0 ft')
ax3.set_ylim(130, 220)

#for sp in speed_val:
#    ax3.axvline(x=sp, color = 'red', linestyle='-', linewidth=1.5)

test_vel = 100.0
ax3.axvline(x=test_vel, color = 'green', linestyle='--', linewidth=1.5, label = 'm/s - kts diff')
ax4.axvline(x=test_vel, color = 'green', linestyle='--', linewidth=1.5)

ax3.grid()
ax3.legend()
fig2.tight_layout()
thr_img_path = os.path.join(img_path, f'Thrust_V_cl_{str.replace(str(cl_best),'.','_')}.pdf')
fig2.savefig(thr_img_path)
print(sep)
print(f'Image Thrust vs. Speed saved in : {thr_img_path}')


#Thrust - altitude depend.
alt_ft = np.arange(0.0, 1500.0, 10.0)
speed_ms = np.arange(50.0, 250.0, 50.0)

fig22 = plt.figure()
for sp in speed_ms:
    thr_alt = np.array([thr_a320.takeoff(tas = conv.convert(sp, 'ms', 'kts'), alt = elev) for elev in alt_ft])

    ax33 = plt.subplot()
    ax33.plot(alt_ft, thr_alt/1000.0, label= f'Thrust (TAS = {sp} m/s)', linestyle = '--')
    ax33.set_xlabel('H [ft]')
    ax33.set_ylabel('Thrust [kN]')
    ax33.set_title(f'Thrust vs. elevation')
    #ax33.set_ylim(10, 250)

ax33.legend()
fig22.tight_layout()
thr_alt_img_path = os.path.join(img_path, f'Thrust_alt.pdf')
fig22.savefig(thr_alt_img_path)
print(sep)
print(f'Image Thrust vs. elevation saved in : {thr_alt_img_path}')

#*************************************************************************************************
#Take-off func test
vel_arr, d_arr, L_arr, D_arr, T_arr, acc_arr, friction_arr, weight_comp_arr = ([] for _ in range(8))

def take_off(m, thrust, rho, cl, cd0, k, w_area, airborne_d, margin_coeff=1.15, 
             mu=0., theta=0., lift_frac=1.0, v_to=74.5, vel_break = False, return_velocity=False, dv0 = 0.01, dv_decay = 'const'):
    vel = 0.01  # m/s
    d = 0.0    # m
    
    theta = conv.convert(theta, 'deg', 'rad')
    cd = cd0 + k * cl* cl
    weight = m * 9.81
   
    #print(f'weight = {weight} N')
    while True:
        # Current state
        D = 0.5 * rho * vel* vel * w_area * cd #parabolic "classic" drag
        '''
        drag = Drag(ac='A320')
        D = drag.clean(mass=m, tas=vel*1.944, alt=0.0, vs=0.0) #OpenAP drag
        '''
        L = 0.5 * rho * vel* vel * w_area * cl
        #print ('init:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        
        if L >= weight * lift_frac:
            break
        if all([vel_break, vel >= v_to]):
            break

        #dv decay selection
        if dv_decay == 'exp': 
            dv = dv0 * np.exp(-L/weight)
        elif dv_decay == 'exp+':
            dv = dv0 * np.exp(-5.0 * L/weight)           
        elif dv_decay == 'inv':
            dv = dv0 * (weight-L) / (L+1.0) 
        elif dv_decay == 'const':
            dv = dv0
        else:
            raise ValueError(f'dv decay type "{dv_decay}" not supported. Try "exp" or "inv" or "const".\nFor further implemention suggestions please contact developers')

        a_current = (thrust - D - mu * (weight * np.cos(theta) - L) - weight * np.sin(theta)) / m

        # Advance velocity
        vel += dv

        # Next state
        D = 0.5 * rho * (vel**2) * w_area * cd
        #D = drag.clean(mass=m, tas=vel*1.944, alt=0.0, vs=0.0) 

        L = 0.5 * rho * (vel**2) * w_area * cl

        a_next = (thrust - D - mu * (weight * np.cos(theta) - L) - weight * np.sin(theta)) / m
        #print ('end:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        #if  a_current <0.0 : print('stupid')
        a_mean = 0.5 * (a_next + a_current)
        if a_mean <= 0.0:
            break  # Prevent division by zero or deceleration

        v_mean = vel - (0.5 * dv)
        dx = v_mean * dv / a_mean
        d += dx
        #print(dv)
        #print(f'v = {vel}, a = {a_mean}. d = {d}')
        vel_arr.append(vel)
        d_arr.append(d)
        L_arr.append(L)
        D_arr.append(D)
        T_arr.append(thrust)
        acc_arr.append(a_current)
        friction_arr.append(mu * (weight * np.cos(theta) - L))
        weight_comp_arr.append(weight)

    final_distance = (d + airborne_d) * margin_coeff
    #print(f'Model take-off vel: {vel} m/s')
    return (final_distance, vel) if return_velocity else final_distance


_ = take_off(m=aircraft_mass, thrust=T[1], rho=rho_isa, cl=cl_best, cd0=cd0, k=k, w_area=wing_area, 
             airborne_d=airborne_dist, vel_break=False, v_to=speed_val[1])


vel_arr = np.array(vel_arr)
d_arr = np.array(d_arr)
L_arr = np.array(L_arr)
D_arr = np.array(D_arr)
T_arr = np.array(T_arr)
acc_arr = np.array(acc_arr)
friction_arr = np.array(friction_arr)
weight_comp_arr = np.array(weight_comp_arr)

# Horizontal forces
fig3= plt.figure(figsize=(10, 6))
ax5 = plt.subplot()
ax6 = ax5.twiny()
ax5.plot(vel_arr, T_arr/1000.0, label='Thrust')
ax6.plot(d_arr, T_arr/1000.0)
ax5.plot(vel_arr, D_arr/1000.0, label='Drag')
ax5.plot(vel_arr, friction_arr/1000.0, label='Friction')
ax5.set_xlabel('V [m/s]')
ax6.set_xlabel('Distance [m]')
ax5.set_ylabel('Horizontal Forces [kN]')
ax5.set_title('Horizontal Forces')
ax5.legend()
#plt.grid(True)
fig3.tight_layout()
save_path = os.path.join(img_path, f'Horiz_forces_{str.replace(str(cl_best),'.','_')}.pdf')
fig3.savefig(save_path)

# Vertical forces
fig4= plt.figure(figsize=(10, 6))
ax7 =  plt.subplot()
ax8 = ax7.twiny()
ax7.plot(vel_arr, L_arr/1000.0, label='Lift')
ax7.plot(vel_arr, weight_comp_arr/1000.0, label='Weight')
ax8.plot(d_arr, L_arr/1000.0, alpha=0)
ax7.set_xlabel('V [m/s]')
ax8.set_xlabel('Distance [m]')
ax7.set_ylabel('Vertical Forces [kN]')
ax7.set_title('Vertical Forces')
ax7.legend()
#plt.grid(True)
fig4.tight_layout()
save_path = os.path.join(img_path, f'Vert_forces_{str.replace(str(cl_best),'.','_')}.pdf')
fig4.savefig(save_path)

# Acceleration vs speed
fig5 = plt.figure(figsize=(10, 6))
ax9 = plt.subplot()
ax9.plot(vel_arr, acc_arr)
ax9.set_xlabel('V [m/s]')
ax9.set_ylabel('Acceleration [m/s^2]')
ax9.set_title('Acceleration vs Velocity')
#plt.grid(True)
fig5.tight_layout()
save_path = os.path.join(img_path, f'acc_vel_{str.replace(str(cl_best),'.','_')}.pdf')
fig5.savefig(save_path)

#*************************************************************************************************
#MTOM calc w/ limited runway lenght
print('===========================================\nMTOM Calculation')
test_runway = np.linspace(1500.0, 2100.0, 13) #m
isa_rho = isa_pr / (r_spec * isa_temp)
print(f'Fixed TODR: {test_runway} m')

#Williams et al. code
start_time = time.monotonic()
mtom_cl = [mtom(lenght, aircraft_mass, T[0], isa_rho, cl_best, cd0, k , wing_area, airborne_dist,
            safe_margin_coef, mu, pathway_incl) for lenght in test_runway]
end_time = time.monotonic()
classic_time = timedelta(seconds = end_time - start_time).total_seconds()

#Binary search
start_time = time.monotonic()
mtom_bin = [mtom_binary(lenght, aircraft_mass, T[0], isa_rho, cl_best, cd0, k , wing_area, airborne_dist,
            safe_margin_coef, mu, pathway_incl, min_mass=50000, tol=1) for lenght in test_runway]
end_time = time.monotonic()
bin_time = timedelta(seconds = end_time - start_time).total_seconds()

print('MTOM calc. results')
for i in range(0,len(test_runway)):
    print(f'TODR[m] = {test_runway[i]}, MTOM_class[kg] = {mtom_cl[i]}, MTOM_bin[kg] = {mtom_bin[i]:.0f}')

print('Computational time:')
print(f'Classic method: {classic_time} s')
print(f'Binary method: {bin_time} s')

'''
#*************************************************************************************************
#Test Williams code (against mine?)
meth_modified = True
cd0_modified = True

if cd0_modified: cd0 = cd0 + mu

mu = 0.02

aircraft_mass = np.array([61235., 63503., 65771., 68039., 70307., 72575., 74843., 77111., 79379.]) #kg
a_mass_err = np.ones(len(aircraft_mass))
to_manuf_value = [1233., 1344., 1455., 1579., 1689., 1798., 1946., 2134., 2362.] # m
to_err = np.ones(len(to_manuf_value))

#Find best C_l values for min, opt and max take-off velocities
cl_values = []
cl_rsmd = []

for i in range(0, len(T)):
    cl_val, err_cl = cl_finder(aircraft_mass, to_manuf_value, to_err, 
                               T[i], rho_isa, cd0, k, wing_area, airborne_dist, safe_margin_coef, v_takeoff=150, mu = mu,
                               dv0=0.01, dv_decay='const', theta = 0.0, cl_min=1.0, cl_max=2.0, cl_step=0.001, modified=meth_modified)

    cl_values.append(cl_val)
    cl_rsmd.append(err_cl)
#print(np.min(cl_values))

#final results
cl_best = np.mean(cl_values)
err_cl_best = 0.5 * (np.max(cl_values) - np.min(cl_values))

print(f"C_l finding process results: C_l = {cl_best} +- {err_cl_best}")
#cl_best = 1.41
#***************************************************************************
#Error analysis

#model prediction and  gt - model perc diff
model_to_dist = np.array([take_off_modified(i, T[1], rho_isa, cl_best, cd0, k, wing_area, airborne_dist, safe_margin_coef, mu=0.02) for i in aircraft_mass])
perc_diff = (model_to_dist - to_manuf_value) / to_manuf_value * 100.0

print('-------------------------------------------------')
print('Perc. difference between Manufacturer and model values:' )
print((perc_diff))
print(f'Mean abs perc. difference: {np.mean(abs(perc_diff)):.3f} %')
print('-------------------------------------------------')
# Upper and lower errors from cl uncertainty
model_upper = np.array([
    take_off_modified(m, T[1], rho_isa, cl_values[0], cd0, k, wing_area, airborne_dist, safe_margin_coef, mu = 0.02)
    for m in aircraft_mass
])
model_lower = np.array([
    take_off_modified(m, T[1], rho_isa, cl_values[2], cd0, k, wing_area, airborne_dist, safe_margin_coef, mu = 0.02)
    for m in aircraft_mass
])
#print(model_lower)
#print(model_to_dist)
#print(model_upper)

#Compute errors
model_err_upper = abs(model_upper - model_to_dist)
model_err_lower = abs(model_to_dist - model_lower)
#print(model_err_lower)
#print(model_err_upper)


#Plots
plt.figure()

# Model with asymmetric error bars
plt.errorbar(aircraft_mass/1000., model_to_dist, 
             yerr=[model_err_lower, model_err_upper],linestyle='',
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
plt.savefig(os.path.join(img_path, f"TODR_mass_stop_mod_{meth_modified}_cd0_mod_{cd0_modified}.pdf"))
'''

# Test parameters
runway_length = 3000    # meters
TODR = 2100            # take-off roll distance [m]
climb_angle_deg = 5.0    # climb angle [°]
grid_scale = 10         # grid extends ±10× runway length

#grid
X,Y,x, y, grid_size =  grid_def(grid_scale, runway_length)

# Generate trajectory
air_x, air_y, air_z = trajectory_s2n(runway_length, grid_size, TODR, climb_angle_deg)

# Plot the trajectory: plan view and altitude profile
fig, (ax0) = plt.subplots(figsize=(12, 5))

# Plan view (X vs Y)
ax0.plot(air_y, air_z)
ax0.set_xlabel('y [m]')
ax0.set_ylabel('z (m)')
ax0.set_title('Trajectory profile')
ax0.grid(True)


plt.savefig(os.path.join(img_path, 'traajectory_test.pdf'))
