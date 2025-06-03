import subprocess
import sys
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv
from file_utils import install_requirements
install_requirements(requirements_file='requirements.txt')
print('---------------------------------------------------------------------------')
from constants import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import time
from datetime import timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning


config_file = 'config.yml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)


# Accessing different sections
cl_path = config['Dir']['cl_dir']
clim_data_dir = config['Dir']['clim_data_dir']
clean_data_dir = config['Dir']['clean_data_dir']

pathway_incl = config['Constants']['pathway_incl']
asc_ft = config['Constants']['asc']
init_climb_angle = config['Constants']['climb_angle']
safe_margin_coef = config['Constants']['margin_coef']

isa_temp = config['ISA']['isa_temp']
isa_pr = config['ISA']['isa_pr']
isa_alt = config['ISA']['isa_alt']

aircraft_name = config['Aircraft']['aircr_name']
engine_name = config['Aircraft']['aircr_engine']
aircraft_mass = config['Aircraft']['aircr_full_m']

airport_code = config['Airport']['airport_code']
airport_length = config['Airport']['airport_lenght']

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

passenger_mass = config['Mass']['passenger_mass']
mass_restr_period = config['Mass']['period']


model_output_path = f'./AP_{airport_code}_AC_{aircraft_name}_{engine_name}'
model_plot_path = model_output_path + '/plots'
performance_parquet_name = f"{airport_code}_{aircraft_name}_{engine_name}_TODR_MTOM.parquet"
performance_parquet_path = os.path.join(model_output_path, performance_parquet_name)

cl_parquet_path = os.path.join(cl_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")

#C_l finder plots
df = pd.read_parquet(cl_parquet_path)
img_out = os.path.join(model_plot_path, f'TODR_mass_{aircraft_name}_{engine_name}.pdf')

plt.figure()

# Model data with asymmetric errors
plt.errorbar(df["mass_tonnes"], df["todr_model"],
             yerr=[df["todr_model_err_upper"], df["todr_model_err_lower"]],
             linestyle='', fmt='', capsize=4, label='Model data', color='tab:blue')

# Manufacturer data with symmetric error
plt.errorbar(df["mass_tonnes"], df["todr_manufacturer"],
             yerr=df["todr_manufacturer_err"], linestyle='--',
            fmt='o', capsize=4, label='Manufacturer data', color='tab:orange')

plt.xlabel('Aircraft mass [* 10^3 kg]')
plt.ylabel('TODR [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(img_out)
# plt.show()  

#*********************************************************************
#TODR and MTOM w/ scenario plot

df_all = pd.read_parquet(performance_parquet_path)
img_name = f'{airport_code}_{aircraft_name}_{engine_name}_TODR_boxplot.pdf'
img_out = os.path.join(model_plot_path, img_name)

#Create the Boxplot
sns.set_style("whitegrid")
#print(img_out)
plt.figure(figsize=(8, 6))

# Define custom order and colors for scenarios if desired
order = ["Historical", "SSP126", "SSP370", "SSP585"]
palette = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c"]

sns.boxplot(x="Scenario", y="TODR", data=df_all, whis =1.5, order=order, palette=palette, showfliers=True)
plt.xlabel("Scenario")
plt.ylabel("TODR [m]")
plt.title(f"TODR (month {climate_months}) -- {aircraft_name} - {engine_name} - {airport_code}")
#sns.despine()
plt.tight_layout()
plt.savefig(img_out)

#---------------------------------------------------------------------------
#TODR distr. hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_TODR_hist.pdf"

# Get the unique scenarios
scenarios = df_all['Scenario'].unique()
n_scenarios = len(scenarios)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['TODR'], bins=N_BINS_TODR, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("TODR [m]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(model_plot_path, hist_name))

    
#temp hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_temp_hist.pdf"

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['mx2t24'], bins=N_BINS_ATM, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("T [K]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.suptitle("Temperature Distribution by Scenario", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(model_plot_path, hist_name))


#sur pres hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_sp_hist.pdf"
    
# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['sp'], bins=N_BINS_ATM, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel("P_sur [Pa]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.suptitle("Surface pressure Distribution by Scenario", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(model_plot_path, hist_name))

#rho pres hist
hist_name = f"{airport_code}_{aircraft_name}_{engine_name}_rho_hist.pdf"

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # To easily index with a single loop

# Plot each scenario in a different panel
for i, scenario in enumerate(scenarios):
    subset = df_all[df_all['Scenario'] == scenario]
    axs[i].hist(subset['rho'], bins=N_BINS_ATM, color='skyblue', edgecolor='black')
    axs[i].set_title(f"{scenario} Scenario")
    axs[i].set_xlabel(r"$\rho$[kg m^-3]")
    axs[i].set_ylabel("Frequency")
    axs[i].grid(True)

plt.suptitle("Air density Distribution by Scenario", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(model_plot_path, hist_name))

print(f'Images saved in {model_plot_path}')

#---------------------------------------------------------------------------
#MTOM and mass restriction 
df_all = pd.read_parquet(performance_parquet_path)
#Compute period means for plot
period_m_restr = (df_all.groupby(['Scenario', mass_restr_period])[['mass_restr_kg', 'mass_restr_pass']]
                  .mean().reset_index()
                  )
mass_restr_scenarios = ["Historical", "SSP126", "SSP370", "SSP585"] #include 'Historical' if needed

fig, axes = plt.subplots(nrows=len(mass_restr_scenarios), ncols=1)

for ax, scenario in zip(axes, mass_restr_scenarios):
    df_s = period_m_restr[period_m_restr['Scenario'] == scenario]

    # Left axis: mass_restr_kg
    ln1 = ax.plot(df_s['Year'], df_s['mass_restr_kg'], linestyle='--',
                    label='Mass Restriction [kg]')
    #ax.grid(True)

    # Right axis: mass_restr_pass
    ax2 = ax.twinx()
    ln2 = ax2.plot(df_s['Year'], df_s['mass_restr_pass'], linestyle='-',
        label='Passenger Restriction [#]')
    
    ax2.set_ylim(-326,-101)
    ax.set_ylim(-326 * passenger_mass, -101 * passenger_mass)
    ax.set_title(f"{scenario} Scenario")

# Common labels
fig.text(0.5, 0.02, 'Year', ha='center')
fig.text(0.02, 0.5, 'Mass restriction [kg]', va='center', rotation='vertical')
fig.text(0.98, 0.5, 'Passenger Restriction [#]', va='center', rotation='vertical')
plt.tight_layout()

mtom_img_name = f'{airport_code}_{aircraft_name}_{engine_name}_MTOM_restr_pass_m{passenger_mass}.pdf'
plt.savefig(os.path.join(model_plot_path, mtom_img_name))

#-----------------------------------------------------------------------------------------
#Hist vs. scenarios comparison
# Compute the Historical‐scenario overall means
hist = period_m_restr[period_m_restr['Scenario'] == 'Historical']
hist_mean_kg   = hist['mass_restr_kg'].mean()
hist_mean_pass = hist['mass_restr_pass'].mean()

# Build a pivot of the SSP scenarios
ssp_scenarios = ['SSP126','SSP370','SSP585']
pivot = period_m_restr.pivot(index='Year', columns='Scenario', values=['mass_restr_kg','mass_restr_pass'])

# Subtract the historical‐mean constant
diff_kg   = pivot['mass_restr_kg'][ssp_scenarios]   - hist_mean_kg
diff_pass = diff_kg / passenger_mass

# Plot
fig, axes = plt.subplots(
    nrows=len(ssp_scenarios), ncols=1)

for ax, scen in zip(axes, ssp_scenarios):
    # Left axis: mass difference
    ln1 = ax.plot(diff_kg.index, diff_kg[scen], linestyle='-',
        label='Δ Mass Restriction [kg]')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    #ax.grid(True)

    # Right axis: passenger difference
    ax2 = ax.twinx()
    ln2 = ax2.plot(diff_pass.index, diff_pass[scen], linestyle='-',
        label='Δ Passenger Restriction [#]')
    ax2.set_ylim(-50, -3)

    
    # "Sync" axes
    ax.set_ylim(-50 *passenger_mass, -3*passenger_mass)
    
    ax.set_title(f"{scen} - additional mass restriction")

# Common  labels
fig.text(0.5, 0.02, 'Year', ha='center')
fig.text(0.02, 0.5, 'Additional Mass restriction [kg]', va='center', rotation='vertical')
fig.text(0.98, 0.5, 'Additional Passenger Restriction [#]', va='center', rotation='vertical')
plt.tight_layout()

mtom_img_name = f'{airport_code}_{aircraft_name}_{engine_name}_ADD_MTOM_restr_pass_m{passenger_mass}.pdf'
plt.savefig(os.path.join(model_plot_path, mtom_img_name))