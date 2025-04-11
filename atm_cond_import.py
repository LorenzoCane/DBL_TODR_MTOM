import xarray as xr
import pandas as pd
import numpy as np
import os
from cmethods import adjust
# Path
airport_code = 'EBBR'
os.makedirs("./data/clean", exist_ok=True) #output dir (clean data)
file_path = f"./data/cmip6_ACCESS-ESM1-5_{airport_code}.nc"
output_path = "./data/clean/"
#Dataset
ds = xr.open_dataset(file_path)
sel_months = [6, 7, 8]

#Some tests
# Print a summary of the dataset
#print(ds)
#print("Data variables", ds.data_vars)

# Specific variable, e.g., air temperature, 
#print(ds['sp_historical'])  # Replace 
#Specific variable values
#print(ds['sp_historical'].values)

#time slicing
#mn2t24 = ds['mn2t24_historical']

#selection = mn2t24.sel(time=slice("1985-01-01", "2014-12-31")).values
#print(selection)

#maximum temperature in the last 24 hours [K]
mx2t24_hist = ds['mx2t24_historical'].sel(time=slice("1985-01-01", "2014-12-31"))
mx2t24_126 =  ds['mx2t24_ssp126'].sel(time=slice("2035-01-01", "2064-12-31"))
mx2t24_370 =  ds['mx2t24_ssp370'].sel(time=slice("2035-01-01", "2064-12-31"))
mx2t24_585 =  ds['mx2t24_ssp585'].sel(time=slice("2035-01-01", "2064-12-31"))

#surface pressure [Pa]
sp_hist = ds['sp_historical'].sel(time=slice("1985-01-01", "2014-12-31"))
sp_126 =  ds['sp_ssp126'].sel(time=slice("2035-01-01", "2064-12-31"))
sp_370 =  ds['sp_ssp370'].sel(time=slice("2035-01-01", "2064-12-31"))
sp_585 =  ds['sp_ssp585'].sel(time=slice("2035-01-01", "2064-12-31"))

#Select months (JJA)
mx2t24_hist_jja = mx2t24_hist.sel(time= mx2t24_hist['time'].dt.month.isin(sel_months))
mx2t24_126_jja =  mx2t24_126.sel(time= mx2t24_126['time'].dt.month.isin(sel_months))
mx2t24_370_jja =  mx2t24_370.sel(time= mx2t24_370['time'].dt.month.isin(sel_months))
mx2t24_585_jja =  mx2t24_585.sel(time= mx2t24_585['time'].dt.month.isin(sel_months))

sp_hist_jja = sp_hist.sel(time= sp_hist['time'].dt.month.isin(sel_months))
sp_126_jja =  sp_126.sel(time= sp_126['time'].dt.month.isin(sel_months))
sp_370_jja =  sp_370.sel(time= sp_370['time'].dt.month.isin(sel_months))
sp_585_jja =  sp_585.sel(time= sp_585['time'].dt.month.isin(sel_months))


def process_scenario(var_temp_name, var_pres_name, time_range, airport, scenario_name, output_path):
    """
    Processes data for a given scenario: selects mx2t24 and sp variables for the given time range,
    filters for JJA months, converts data to a DataFrame with extra columns,
    drops any rows where mx2t24 or sp are NaN, then saves the DataFrame to a CSV file.
    """
    # Select variables within the time range.
    temp_var = ds[var_temp_name].sel(time=slice(time_range[0], time_range[1]))
    pres_var = ds[var_pres_name].sel(time=slice(time_range[0], time_range[1]))
    
    # Filter only for JJA months.
    temp_jja = temp_var.sel(time=temp_var['time'].dt.month.isin(sel_months))
    pres_jja = pres_var.sel(time=pres_var['time'].dt.month.isin(sel_months))
    
    # Merge both variables into one dataset (they share the same time and variant dimensions).
    ds_merged = xr.merge([temp_jja, pres_jja])
    
    # Convert to a DataFrame (reset_index turns coordinates like time and variant into columns).
    df = ds_merged.to_dataframe().reset_index()
    
    # Add extra columns: Scenario, Year, Month, Day.
    df['Scenario'] = scenario_name
    df['time'] = pd.to_datetime(df['time'])
    df['Year'] = df['time'].dt.year
    df['Month'] = df['time'].dt.month
    df['Day'] = df['time'].dt.day
    
    # Rename variables for clarity (e.g., 'mx2t24_historical' becomes 'mx2t24').
    df = df.rename(columns={
        var_temp_name: 'mx2t24',
        var_pres_name: 'sp'
    })

    # Perform 3-sigma filtering for both 'mx2t24' and 'sp' columns.
    # Calculate the mean and standard deviation for both variables.
    temp_mean = df['mx2t24'].mean()
    temp_std = df['mx2t24'].std()
    pres_mean = df['sp'].mean()
    pres_std = df['sp'].std()
    
    # Define the bounds for 3-sigma filtering.
    temp_lower_bound = temp_mean - 3 * temp_std
    temp_upper_bound = temp_mean + 3 * temp_std
    pres_lower_bound = pres_mean - 3 * pres_std
    pres_upper_bound = pres_mean + 3 * pres_std
    
    # Apply the 3-sigma filtering for both 'mx2t24' and 'sp' columns.
    df = df[(df['mx2t24'] >= temp_lower_bound) & (df['mx2t24'] <= temp_upper_bound)]
    df = df[(df['sp'] >= pres_lower_bound) & (df['sp'] <= pres_upper_bound)]
    
    # Order the desired columns: Scenario, time, variant (if exists), mx2t24, sp, Year, Month, Day.
    desired_columns = ['Scenario', 'time', 'variant', 'mx2t24', 'sp', 'Year', 'Month', 'Day']
    df = df[[col for col in desired_columns if col in df.columns]]
    
    # Drop rows where either 'mx2t24' or 'sp' is NaN.
    df = df.dropna(subset=['mx2t24', 'sp'])
    
    # Define output filename based on the scenario.
    output_filename = f"{airport}_{scenario_name}_JJA.csv"
    output_path = os.path.join(output_path, output_filename)
    # Save to CSV.
    df.to_csv(output_path, index=False)
    print(f"Data for {airport}_{scenario_name} scenario saved to {output_path}")


# Process Historical data (1985-2014)
df_hist = process_scenario(
    var_temp_name='mx2t24_historical',
    var_pres_name='sp_historical',
    time_range=("1985-01-01", "2014-12-31"),
    airport= airport_code,
    scenario_name="Historical",
    output_path=output_path
)

# Process SSP scenarios (2035-2064)
df_ssp126 = process_scenario(
    var_temp_name='mx2t24_ssp126',
    var_pres_name='sp_ssp126',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code,
    scenario_name="SSP126",
    output_path=output_path
)
df_ssp370 = process_scenario(
    var_temp_name='mx2t24_ssp370',
    var_pres_name='sp_ssp370',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code,
    scenario_name="SSP370",
    output_path=output_path
)
df_ssp585 = process_scenario(
    var_temp_name='mx2t24_ssp585',
    var_pres_name='sp_ssp585',
    time_range=("2035-01-01", "2064-12-31"),
    airport= airport_code, 
    scenario_name="SSP585",
    output_path=output_path
)

#******************************************************************************

def quantile_delta_mapping(obs, sim_hist, sim_fut):
    """
    Applies Quantile Delta Mapping (QDM) correction.
    All inputs are numpy arrays (1D).
    """
    qsim_hist = np.quantile(sim_hist, np.linspace(0, 1, len(sim_hist)))
    qobs = np.quantile(obs, np.linspace(0, 1, len(obs)))
    qsim_fut = np.quantile(sim_fut, np.linspace(0, 1, len(sim_fut)))
    delta = qsim_fut - qsim_hist
    corrected = qobs + delta
    return corrected

def process_scenario_qdm(var_temp_name, var_pres_name, time_range, airport, scenario_name, output_path):
    """
    Processes and applies QDM correction to temperature and pressure data, saving a cleaned CSV.
    """
    # Historical reference for QDM
    temp_hist_ref = ds['mx2t24_historical'].sel(time=slice("1985-01-01", "2014-12-31"))
    pres_hist_ref = ds['sp_historical'].sel(time=slice("1985-01-01", "2014-12-31"))
    temp_hist_ref_jja = temp_hist_ref.sel(time=temp_hist_ref['time'].dt.month.isin(sel_months)).values.flatten()
    pres_hist_ref_jja = pres_hist_ref.sel(time=pres_hist_ref['time'].dt.month.isin(sel_months)).values.flatten()

    # Current scenario
    temp_var = ds[var_temp_name].sel(time=slice(time_range[0], time_range[1]))
    pres_var = ds[var_pres_name].sel(time=slice(time_range[0], time_range[1]))
    temp_jja = temp_var.sel(time=temp_var['time'].dt.month.isin(sel_months))
    pres_jja = pres_var.sel(time=pres_var['time'].dt.month.isin(sel_months))

    # Flatten values for QDM
    temp_sim_hist = temp_hist_ref_jja
    pres_sim_hist = pres_hist_ref_jja
    temp_sim_fut = temp_jja.values.flatten()
    pres_sim_fut = pres_jja.values.flatten()

    # Apply QDM correction
    temp_corrected = quantile_delta_mapping(temp_hist_ref_jja, temp_sim_hist, temp_sim_fut)
    pres_corrected = quantile_delta_mapping(pres_hist_ref_jja, pres_sim_hist, pres_sim_fut)

    # Reshape back to original shape
    temp_jja_corrected = temp_jja.copy(data=temp_corrected.reshape(temp_jja.shape))
    pres_jja_corrected = pres_jja.copy(data=pres_corrected.reshape(pres_jja.shape))

    # Merge and convert to DataFrame
    ds_merged = xr.merge([temp_jja_corrected.rename("mx2t24"), pres_jja_corrected.rename("sp")])
    df = ds_merged.to_dataframe().reset_index()
    df['Scenario'] = scenario_name
    df['time'] = pd.to_datetime(df['time'])
    df['Year'] = df['time'].dt.year
    df['Month'] = df['time'].dt.month
    df['Day'] = df['time'].dt.day

    # Drop missing values
    df = df.dropna(subset=['mx2t24', 'sp'])

    # Perform 3-sigma filtering for both 'mx2t24' and 'sp' columns.
    # Calculate the mean and standard deviation for both variables.
    temp_mean = df['mx2t24'].mean()
    temp_std = df['mx2t24'].std()
    pres_mean = df['sp'].mean()
    pres_std = df['sp'].std()
    
    # Define the bounds for 3-sigma filtering.
    temp_lower_bound = temp_mean - 3 * temp_std
    temp_upper_bound = temp_mean + 3 * temp_std
    pres_lower_bound = pres_mean - 3 * pres_std
    pres_upper_bound = pres_mean + 3 * pres_std
    
    # Apply the 3-sigma filtering for both 'mx2t24' and 'sp' columns.
    df = df[(df['mx2t24'] >= temp_lower_bound) & (df['mx2t24'] <= temp_upper_bound)]
    df = df[(df['sp'] >= pres_lower_bound) & (df['sp'] <= pres_upper_bound)]
    # Column order
    desired_columns = ['Scenario', 'time', 'variant', 'mx2t24', 'sp', 'Year', 'Month', 'Day']
    df = df[[col for col in desired_columns if col in df.columns]]

    # Save to CSV
    output_filename = f"{airport}_{scenario_name}_QDM_JJA.csv"
    df.to_csv(os.path.join(output_path, output_filename), index=False)
    print(f"QDM-corrected data for {scenario_name} saved to {output_path + output_filename}")

# Apply QDM and save
process_scenario_qdm('mx2t24_ssp126', 'sp_ssp126', ("2035-01-01", "2064-12-31"), airport_code, "SSP126", output_path)
process_scenario_qdm('mx2t24_ssp370', 'sp_ssp370', ("2035-01-01", "2064-12-31"), airport_code, "SSP370", output_path)
process_scenario_qdm('mx2t24_ssp585', 'sp_ssp585', ("2035-01-01", "2064-12-31"), airport_code, "SSP585", output_path)
process_scenario_qdm('mx2t24_historical', 'sp_historical', ("1985-01-01", "2014-12-31"), airport_code, "Historical", output_path)