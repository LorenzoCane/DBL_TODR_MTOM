import numpy as np
import os
import sys

from openap import prop #aircraft and engine-related data
from openap.kinematic import WRAP #set of kinematic models
from openap.thrust import Thrust #thrust calc
from openap.drag import Drag
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv

def get_aircraft_data(aircraft_name, engine_name):
    #aircraft specif.
    aircraft = prop.aircraft(aircraft_name) #airbus A320
    engine = prop.engine(engine_name) #V2500-A1 turbofan engines
    wing_area = aircraft['wing']['area'] #wing area
    cd0 = aircraft['drag']['cd0']
    k = aircraft['drag']['k']
    mu = aircraft['drag']['gears']
    full_mass =  aircraft['mtow']

    useful_dict = {
        "wing_area": wing_area,
        "cd0": cd0,
        "k": k,
        "mu": mu,
        "mass_max": full_mass
    }

    return useful_dict

def get_to_speed(aircraft_name):
    #aircraft TO speeds
    wrap = WRAP(ac=aircraft_name) #kinematic parameters
    to_speed = wrap.takeoff_speed() # m/s Take-off speed. order: default (optimum), minimum, maximum
    '''
    opt_to_speed = to_speed['default'] #m/s
    min_to_speed = to_speed['minimum'] #m/s
    max_to_speed = to_speed['maximum'] #m/s
    '''
    speed_val = np.sort([s for s in list(to_speed.values())[:3]])

    return speed_val

#***************************************************************************
def get_thrust(aircraft_name, engine_name, alt_m):
    to_speed = get_to_speed(aircraft_name)

    thr = Thrust(ac= aircraft_name, eng= engine_name)
    T = np.array([thr.takeoff(tas = conv.convert(sp, 'ms', 'kts'), alt=conv.convert(alt_m, 'm', 'ft')) for sp in to_speed]) #N conv

    return T

#***************************************************************************
def manuf_data_loader(cl_path, aircraft_name, engine_name):
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

    return aircraft_mass, to_manuf_value, to_err
