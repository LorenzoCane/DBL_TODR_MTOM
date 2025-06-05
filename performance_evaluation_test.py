import pyarrow.parquet as pq
from airport_utils import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from aircraft_utils import *
from config_loader import load_config
from todr import take_off, mtom, mtom_binary
sys.path.insert(0,'./utils')
from unit_converter import ComplexUnitConverter as conv

def process_row(rho_val, mass_max, thr, cl_best, cd0, k, wing_area,
                airborne_dist, MARGIN_COEFF, mu, INCL, runway_length,
                passenger_mass):
    '''
     Compute aircraft permonce in parallelize way (still testing)
    '''
    # Compute TODR
    todr = take_off(mass_max, thr, rho_val, cl_best, cd0, k, wing_area,
                    airborne_dist, MARGIN_COEFF, mu, INCL)
    # Compute MTOM
    mtom_val = mtom_binary(runway_length, mass_max, thr, rho_val, cl_best, cd0, k,
                           wing_area, airborne_dist, MARGIN_COEFF, mu, INCL)
    # Mass restriction in kg and passengers
    mass_restr = mtom_val - mass_max
    mass_restr_pass = mass_restr // passenger_mass
    return todr, mtom_val, mass_restr, mass_restr_pass

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
    print(f"Processed scenario: {scenario} ({len(df)} rows)")
    # Compute air density: rho = Pressure / (R * Temperature)
    df["rho"] = df["sp"] / (R_SPEC * df["mx2t24"])

    # Parallelized per-row calc
    args_fixed = (mass_max, thr, cl_best, cd0, k, wing_area,
              airborne_dist, MARGIN_COEFF, mu, INCL,
              airport_l_m, passenger_mass)
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(partial(process_row, *args_fixed), df["rho"].values))

    #Save results in df rows (+ check against dim failure)
    if len(results) == len(df):
        df["TODR"], df["MTOM"], df["mass_restr_kg"], df["mass_restr_pass"] = zip(*results)
    else:
        raise ValueError("Mismatch in result lengths. Check for failures in process_row.")


    # Remove rows where TODR, temperature, or pressure are NaN
    #df = df.dropna(subset=["TODR", "mx2t24", "sp"])
    df["Scenario"] = scenario  # <-- add this before appending
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



