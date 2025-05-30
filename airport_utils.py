import airportsdata #airport data
import os
import pandas as pd
from config_loader import load_config
from constants import *

config = load_config()
clim_data_dir = config['Dir']['clim_data_dir']
airports = airportsdata.load()

def airport_get_lat(airport_code):
    sel_airport = airports[airport_code]
    return sel_airport['lat']

def airport_get_log(airport_code):
    sel_airport = airports[airport_code]
    return sel_airport['lon']

def airport_get_elev(airport_code):
    sel_airport = airports[airport_code]
    return sel_airport['elevation']

def airport_get_name(airport_code):
    sel_airport = airports[airport_code]
    return sel_airport['name']

def airport_get_lenght(airport_code):
    df_airport = pd.read_csv(os.path.join(clim_data_dir, "airport_runways.csv"))
    airport_length_m = df_airport.loc[df_airport['ICAO'] == airport_code, "Runway_Length_m"].values[0] #m
    if airport_length_m < 0 : 
        raise ValueError(f'ICAO code {airport_code} not found.')
    return airport_length_m