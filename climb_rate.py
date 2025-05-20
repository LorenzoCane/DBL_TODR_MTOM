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


#***************************************************************************
#import from configuration file config.yml
config_file = 'config.yml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

img_path = config['Dir']['img_dir']
output_path =  config['Dir']['output_dir']
clim_data_dir = config['Dir']['clim_data_dir']

airport_code = config['Airport']['airport_code']

aircraft_name = config['Aircraft']['aircr_name']
engine_name = config['Aircraft']['aircr_engine']

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

isa_temp = config['ISA']['isa_temp']
isa_pr = config['ISA']['isa_pr']
isa_alt = config['ISA']['isa_alt']

print(f'Configuration successfully loaded from {config_file}')
#***************************************************************************
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
#---------------------------------------------------------------------------
#airport data

#TO BE DONE


#---------------------------------------------------------------------------
#Constants

G = 9.81
R_SPEC = 287.0528

#***************************************************************************
#***************************************************************************
#Code scheme
'''
lenght from airport data
Clim analisis (no MTOM necess.) -> if TODR > runway_l impose runway_l

noise grid (should be autom)
rotate and add to map (how)

'''
#***************************************************************************
#***************************************************************************