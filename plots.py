
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning


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

print(f'Configuration successfully loaded from {config_file}')
print('-------------------------------------------------')

cl_file_name = 'cl_TODR_data.parquet'
QDM_file_name =  f"{airport_code}_TODR_QDM.parquet"
NOQDM_file_name =  f"{airport_code}_TODR_NOQDM.parquet"

cl_data_path = os.path.join(output_path, cl_file_name)
QDM_path = os.path.join(output_path, QDM_file_name)
NOQDM_path = os.path.join(output_path, NOQDM_file_name)
files = [QDM_path, NOQDM_path]
os.makedirs(img_path, exist_ok=True)
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

for file, id in zip(files, ['QDM', 'NOQDM']):
    df_all = pd.read_parquet(file)
    img_name = f'{airport_code}_{id}.pdf'
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
    n_bins = 30
    #TODR distr. hist
    hist_name = f"{airport_code}__TODR_{id}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['TODR'], bins=n_bins, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("TODR")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

    n_bins = 50
    #temp hist
    hist_name = f"{airport_code}__temp_{id}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['mx2t24'], bins=n_bins, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("T [K]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))


    #sur pres hist
    hist_name = f"{airport_code}__sp_{id}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['sp'], bins=n_bins, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("sp [Pa]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

    #rho pres hist
    hist_name = f"{airport_code}__rho_{id}_hist.pdf"
    # Get the unique scenarios
    scenarios = df_all['Scenario'].unique()
    n_scenarios = len(scenarios)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # To easily index with a single loop

    # Plot each scenario in a different panel
    for i, scenario in enumerate(scenarios):
        subset = df_all[df_all['Scenario'] == scenario]
        axs[i].hist(subset['rho'], bins=n_bins, color='skyblue', edgecolor='black')
        axs[i].set_title(f"{scenario} Scenario")
        axs[i].set_xlabel("air density [kg m^-3]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle("TODR Distribution by Scenario", fontsize=16, y=1.02)
    plt.savefig(os.path.join(img_path, hist_name))

print(f'Images saved in {img_path}')