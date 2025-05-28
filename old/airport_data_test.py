
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
from take_off_func import take_off, cl_finder

import airportsdata


#Import airports data 'IATA' or 'FAA LID' location can be used 
airports = airportsdata.load()  # key is the ICAO identifier (the default) 

#Structure: airports['ICAO_code']['key']
#example

selected_airport = airports['LIPE'] #LIPE = Bologna Borgo Panigale
airport_name = selected_airport['name']
lat = selected_airport['lat']
long = selected_airport['lon']
height = selected_airport['elevation'] #MSL elevation of the highest point of the landing area, in feet (warning: it is often wrong);

print(f'Ex.: {airport_name}:')
print(f'Lat: {lat}, Long: {long}, Elevation (MSL): {height} ft')


