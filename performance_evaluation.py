import pyarrow.parquet as pq
from airport_utils import *
from aircraft_utils import *
from config_loader import load_config
from todr import take_off, mtom
sys.path.insert(0,'./utils')
from unit_converter import ComplexUnitConverter as conv

config = load_config()

cl_path = config['Dir']['cl_dir']
clean_data_dir = config['Dir']['clean_data_dir']

aircraft_name = config["Aircraft"]["aircr_name"]
engine_name = config["Aircraft"]["aircr_engine"]
airport_code = config['Airport']['airport_code']
passenger_mass = config['Mass']['passenger_mass']


aircraft_info = get_aircraft_data(aircraft_name, engine_name)

wing_area = aircraft_info["wing_area"]
cd0 = aircraft_info["cd0"]
k = aircraft_info["k"]
mu = aircraft_info["mu"]
mass_max = aircraft_info["mass_max"]

asc_m = conv.convert(ASC, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(CLIMB_ANGLE_DEG, 'deg', 'rad')) # m

rho_isa = ISA_PR / (R_SPEC * ISA_TEMP)

airport_l_m = airport_get_lenght(airport_code)

model_output_path = f'./AP_{airport_code}_AC_{aircraft_name}_{engine_name}'
model_plot_path = model_output_path + '/plots'

cl_parquet_path = os.path.join(cl_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")
# Load the Parquet file with C_L value
table = pq.read_table(cl_parquet_path)
# Extract and decode metadata
metadata = table.schema.metadata
decoded_meta = {k.decode(): v.decode() for k, v in metadata.items()}
cl_best = float(decoded_meta["cl_best"])
thr = float(decoded_meta["T_used"])

print(f"C_l best: {cl_best}")


#Re-read csv files
file_dict = {
    "Historical": airport_code + "_Historical_JJA.csv",
    "SSP126": airport_code + "_SSP126_JJA.csv",
    "SSP370": airport_code + "_SSP370_JJA.csv",
    "SSP585": airport_code + "_SSP585_JJA.csv"
}


all_data = []  # list to hold each scenario's processed DataFrame
for scenario, filename in file_dict.items():
    # Read the CSV; each file should have at least columns: "mx2t24" (temperature, [K]) and "sp" (pressure, [Pa])
    df = pd.read_csv(os.path.join(clean_data_dir, filename))
    print(f'Creating DataFrame for scenario: {scenario}')
    # Compute air density: rho = Pressure / (R * Temperature)
    df["rho"] = df["sp"] / (R_SPEC * df["mx2t24"])
    print("Air densities evaluated")
    # Compute TODR for each row using the same constant parameters
    df["TODR"] = df.apply(lambda row: take_off(mass_max, thr, row["rho"], cl_best, cd0, k, wing_area, 
                                               airborne_dist, MARGIN_COEFF, mu, INCL), axis=1)
    print("TODR evaluated")
    # Compute MTOM for each row
    df["MTOM"] = df.apply(lambda row: mtom(airport_l_m, mass_max, thr, row["rho"], cl_best, cd0, k,
                                                  wing_area, airborne_dist, MARGIN_COEFF, mu, INCL), axis = 1)
    print("MTOM evaluated")
    # Compute mass reduction in kg and n. of passenger
    df["mass_restr_kg"] = df["MTOM"] - mass_max #kg Negative numbers
    
    df["mass_restr_pass"] = df['mass_restr_kg'] // passenger_mass #being neg counts one more "cancelled passanger" (conservative way)
    print('Mass restriction evaluated')
    # Add a column for the scenario label
    df["Scenario"] = scenario
    print(sep)
    # Remove rows where TODR, temperature, or pressure are NaN
    #df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    all_data.append(df)

# Concatenate all the scenario DataFrames
df_all = pd.concat(all_data, ignore_index=True)
print("\n=== TODR Summary by Scenario ===")
print(df_all.groupby("Scenario")["TODR"].describe().round(2))

#Save data for future plots and anal
performance_parquet_name = f"{airport_code}_{aircraft_name}_{engine_name}_TODR_MTOM.parquet"
performance_parquet_path = os.path.join(model_output_path, performance_parquet_name)

df_all.to_parquet(performance_parquet_path)
print(f'All processed data saved in {performance_parquet_path}')



