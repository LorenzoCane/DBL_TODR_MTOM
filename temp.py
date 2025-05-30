import os

# 1. Create folders structure
folders = [
    "src",
    "src/utils",
    "src/plots",
    "data/raw",
    "data/processed",
    "outputs/plots",
    "outputs/results"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 2. Create src/config_loader.py
with open("src/config_loader.py", "w") as f:
    f.write('''
import yaml

def load_config(path='config.yml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
''')

# 3. Create src/constants.py
with open("src/constants.py", "w") as f:
    f.write('''
G = 9.81  # m/s^2
R_SPEC = 287.0528

# Plot settings
N_BINS_TODR = 50
N_BINS_ATM = 100
''')