import xarray as xr
import pandas as pd
import numpy as np
import os




def process_scenario(ds_select, var_temp_name, var_pres_name, time_range, airport, scenario_name, output_path, sel_months = [ 6, 7, 8 ]):
    """
    Processes data for a given scenario: selects mx2t24 and sp variables for the given time range,
    filters for JJA months, converts data to a DataFrame with extra columns,
    drops any rows where mx2t24 or sp are NaN, then saves the DataFrame to a CSV file.
    """
    # Select variables within the time range.
    temp_var = ds_select[var_temp_name].sel(time=slice(time_range[0], time_range[1]))
    pres_var = ds_select[var_pres_name].sel(time=slice(time_range[0], time_range[1]))
    
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
    
    '''
    # Define the bounds for 3-sigma filtering.
    temp_lower_bound = temp_mean - 3 * temp_std
    temp_upper_bound = temp_mean + 3 * temp_std
    pres_lower_bound = pres_mean - 3 * pres_std
    pres_upper_bound = pres_mean + 3 * pres_std
    
    # Apply the 3-sigma filtering for both 'mx2t24' and 'sp' columns.
    df = df[(df['mx2t24'] >= temp_lower_bound) & (df['mx2t24'] <= temp_upper_bound)]
    df = df[(df['sp'] >= pres_lower_bound) & (df['sp'] <= pres_upper_bound)]
    '''
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