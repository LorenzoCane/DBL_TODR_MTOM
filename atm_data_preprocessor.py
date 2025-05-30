import numpy as np
import xarray as xr
import os
from config_loader import load_config
from constants import *
from atm_anal_func import process_scenario

config = load_config()

cl_path = config['Dir']['cl_dir']
clim_data_dir = config['Dir']['clim_data_dir']
clean_data_dir = config['Dir']['clean_data_dir']

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

airport_code = config['Airport']['airport_code']

#Atm data 
file_path = os.path.join(clim_data_dir , f"cmip6_{climate_model}_{airport_code}.nc")
#Dataset
ds = xr.open_dataset(file_path)
sel_months = climate_months


# Process Historical data (1985-2014)
df_hist = process_scenario(
    ds,
    var_temp_name='mx2t24_historical',
    var_pres_name='sp_historical',
    time_range=("1985-01-01", "2014-12-31"),
    airport= airport_code,
    scenario_name="Historical",
    output_path=clean_data_dir,
    sel_months= climate_months
)

# Process SSP scenarios (2035-2064)
df_ssp126 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp126',
    var_pres_name='sp_ssp126',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code,
    scenario_name="SSP126",
    output_path=clean_data_dir,
    sel_months= climate_months
)
df_ssp370 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp370',
    var_pres_name='sp_ssp370',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code,
    scenario_name="SSP370",
    output_path=clean_data_dir,
    sel_months= climate_months
)
df_ssp585 = process_scenario(
    ds,
    var_temp_name='mx2t24_ssp585',
    var_pres_name='sp_ssp585',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code, 
    scenario_name="SSP585",
    output_path=clean_data_dir,
    sel_months= climate_months
)