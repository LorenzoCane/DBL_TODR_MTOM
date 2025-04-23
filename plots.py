
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import git
import yaml
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning

import files_manager_utils as fmu
#***************************************************************************
#import from configuration file config.yml
config_file = 'config.yml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)


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
#airport_alt_m = config['Airport']['airport_alt']
#airport_alt_ft = conv.convert(airport_alt_m, 'm', 'ft')

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

dv0 = config['Speed']['v0']
dv_decay = config['Speed']['decay']

print(f'Configuration successfully loaded from {config_file}')
print('-------------------------------------------------')

cl_file_name = 'cl_TODR_data.parquet'
QDM_file_name =  f"{airport_code}_TODR_QDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}.parquet"
NOQDM_file_name = f"{airport_code}_TODR_NOQDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}.parquet"
'''
cl_file_name = 'cl_TODR_data_vel_break.parquet'
QDM_file_name =  f"{airport_code}_TODR_QDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_vel_break.parquet"
NOQDM_file_name = f"{airport_code}_TODR_NOQDM_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_vel_break.parquet"
'''

cl_data_path = os.path.join(output_path, cl_file_name)
QDM_path = os.path.join(output_path, QDM_file_name)
NOQDM_path = os.path.join(output_path, NOQDM_file_name)
files = [NOQDM_path]
#files = [QDM_path, NOQDM_path]
os.makedirs(img_path, exist_ok=True)

#plots settings
n_bins_todr = 50
n_bins_atm = 100

temp_min = 300.0 #K lower bound for temp mask (100 --> no lower bound)
temp_max = 350.0 #K upper bound for temp mask (700 --> no upper bound)

#*********************************************************************
#creation of folder(s) with progressive names 
#TO RESTART THE COUNTING: delete Flag_variable.npy file from results folder

 #search for the current number of results
f=os.path.join(img_path, 'Flag_variable.npy')
try:
    Flag_variable = np.load(os.path.join(img_path, 'Flag_variable.npy'))
    Flag_variable = int(Flag_variable)
    Flag_variable =Flag_variable+ 1
    np.save(f,Flag_variable)

except FileNotFoundError:
    
    np.save(f, 1)
    Flag_variable=1
#create a new folder    
Newfolder= f'results_{str(Flag_variable)}_dv{dv0}_{dv_decay}'
img_path = os.path.join(img_path, Newfolder)
os.makedirs(img_path, exist_ok=True)

print(f"New folder: {img_path} has been created")

#copy config.yml file into the results 
source_file = "config.yml"
new_file_name = "config" + str(Flag_variable) + ".yml"
fmu.copy_and_rename(source_file, img_path, new_file_name) #include files_manager_utils.py in your import
#print current files version in the config file footer
f = open(os.path.join(img_path,new_file_name), "a")
f.write("\n\n#*************************************\n")
f.write("#Last git commit:\n")
repo = git.Repo("./")   
tree = repo.tree()
for blob in tree:
    commit = next(repo.iter_commits(paths=blob.path, max_count=1))
    version = "# " + str(blob.path) + " : " + str(commit.committed_date) + "\n"
    f.write(version)

f.close()
#*********************************************************************
'''
#Define path
airport_code = 'EBBR'
data_dir = './output_data/'

cl_file_name = 'cl_TODR_data.parquet'
QDM_file_name =  f"{airport_code}_TODR_QDM.parquet"
NOQDM_file_name =  f"{airport_code}_TODR_NOQDM.parquet"

cl_data_path = os.path.join(data_dir, cl_file_name)
QDM_path = os.path.join(data_dir, QDM_file_name)
NOQDM_path = os.path.join(data_dir, NOQDM_file_name)

files = [QDM_path, NOQDM_path]
os.makedirs('images', exist_ok=True)
img_path = './images/'
'''

#*********************************************************************
#C_l finder plots
df = pd.read_parquet(cl_data_path)
img_out = os.path.join(img_path, 'TODR_mass.pdf')

plt.figure()

# Model data with asymmetric errors
plt.errorbar(df["mass_tonnes"], df["todr_model"],
             yerr=[df["todr_model_err_upper"], df["todr_model_err_lower"]],
             linestyle='', fmt='', capsize=4, label='Model data', color='tab:blue')

# Manufacturer data with symmetric error
plt.errorbar(df["mass_tonnes"], df["todr_manufacturer"],
             yerr=df["todr_manufacturer_err"], linestyle='--',
            fmt='o', capsize=4, label='Manufacturer data', color='tab:orange')

plt.xlabel('Aircraft mass [10^3 kg]')
plt.ylabel('TODR [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(img_out)
# plt.show()  

#*********************************************************************
#TODR and MTOM w/ scenario plot

for file, id in zip(files, ['NOQDM']): 
    df_all = pd.read_parquet(file)
    img_name = f'{airport_code}_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}.pdf'

    #img_name = f'{airport_code}_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_vel_break.pdf'
    img_out = os.path.join(img_path, img_name)
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
    plt.title(f"TODR (JJA) {aircraft_name} - {engine_name} - {airport_code} - {id}")
    #sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(img_path, img_name))

    #---------------------------------------------------------------------------
    #TODR distr. hist
    hist_name = f"{airport_code}_TODR_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"

    #hist_name = f"{airport_code}_TODR_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist_v_br.pdf"

    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['TODR'], bins=n_bins_todr, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("TODR [m]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

    
    #temp hist
    hist_name = f"{airport_code}_temp_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"

    #hist_name = f"{airport_code}_temp_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist_vel_break.pdf"

    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['mx2t24'], bins=n_bins_atm, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("T [K]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))


    #sur pres hist
    hist_name = f"{airport_code}_sp_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"
    
    #hist_name = f"{airport_code}_sp_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist_vel_break.pdf"

    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['sp'], bins=n_bins_atm, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("sp [Pa]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

    #rho pres hist
    hist_name = f"{airport_code}_rho_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"
    
    #hist_name = f"{airport_code}_rho_{id}_dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist_vel_break.pdf"

    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['rho'], bins=n_bins_atm, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("air density [kg m^-3]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

print(f'Images saved in {img_path}')


'''
#*********************************************************************
#TODR and atm cond w/ temp bounds

for file, id in zip(files, ['QDM', 'NOQDM']):
    df_all = pd.read_parquet(file)
    
    #---------------------------------------------------------------------------
    #TODR distr. hist
    hist_name = f"{airport_code}_TODR_{id}_t_{temp_min}_{temp_max}__dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[(df_all['Scenario'] == scenario) &
                        (df_all['mx2t24'] > temp_min) &
                        (df_all['mx2t24'] < temp_max)
                        ]
        axs[i].hist(subset['TODR'], bins=n_bins_todr, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("TODR")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

    
    #temp hist
    hist_name = f"{airport_code}_temp_{id}_t_{temp_min}_{temp_max}__dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[(df_all['Scenario'] == scenario) &
                        (df_all['mx2t24'] > temp_min) &
                        (df_all['mx2t24'] < temp_max)
                        ]
        axs[i].hist(subset['mx2t24'], bins=n_bins_atm, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("T [K]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))


    #sur pres hist
    hist_name = f"{airport_code}_sp_{id}_t_{temp_min}_{temp_max}__dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[(df_all['Scenario'] == scenario) &
                        (df_all['mx2t24'] > temp_min) &
                        (df_all['mx2t24'] < temp_max)
                        ]
        axs[i].hist(subset['sp'], bins=n_bins_atm, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("sp [Pa]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

    #rho pres hist
    hist_name = f"{airport_code}_rho_{id}_t_{temp_min}_{temp_max}__dv_{str(dv0).replace( '.' , '_' )}_{dv_decay}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[(df_all['Scenario'] == scenario) &
                        (df_all['mx2t24'] > temp_min) &
                        (df_all['mx2t24'] < temp_max)
                        ]
        axs[i].hist(subset['rho'], bins=n_bins_atm, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("air density [kg m^-3]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

print(f'Images saved in {img_path}')

'''