import numpy as np
import subprocess
import threading
import time
import sys
from tqdm import tqdm


class ComplexUnitConverter:
    CONVERSION_TABLE = {
        ('lb', 'kg'): 0.453592,
        ('kg', 'lb'): 2.20462442,
        ('inch', 'm'): 0.0254,
        ('m', 'inch'): 39.37,
        ('ft', 'm'): 0.3048,
        ('m', 'ft'): 3.280839995,
        ('ms', 'kts'): 1.94,
        ('kts', 'ms'): 0.51444563,
        ('deg', 'rad'): np.pi / 180.0,
        ('rad', 'deg'): 180.0/ np.pi,
        ('celsius', 'fahrenheit'): lambda celsius: celsius * 9 / 5 + 32,
        ('fahrenheit', 'celsius'): lambda fahrenheit: (fahrenheit - 32) * 5 / 9,
        ('celsius', 'kelvin'): lambda celsius: (celsius + 273.15)
    }

    @classmethod
    def convert(cls, value, from_unit, to_unit):
        # Handle direct conversions from the table
        if (from_unit, to_unit) in cls.CONVERSION_TABLE:
            conversion_factor = cls.CONVERSION_TABLE[(from_unit, to_unit)]
            
            # If the conversion factor is a lambda function, apply it
            if callable(conversion_factor):
                return conversion_factor(value)
            else:
                return value * conversion_factor
        
        # Handle reverse conversions (from to to_unit to from_unit)
        elif (to_unit, from_unit) in cls.CONVERSION_TABLE:
            conversion_factor = cls.CONVERSION_TABLE[(to_unit, from_unit)]
            
            # If the conversion factor is a lambda function, apply it
            if callable(conversion_factor):
                return conversion_factor(value)
            else:
                return value / conversion_factor
        
        # If the conversion is not in the table, raise an error
        raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")

#***************************************************************************

def install_requirements(requirements_file="requirements.txt"):
    print(f'Installing requirements (see {requirements_file})')
    def pip_install():
        try:
            # Suppress output with -q (quiet)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            status[0] = "done"
        except subprocess.CalledProcessError:
            status[0] = "error"
        except FileNotFoundError:
            status[0] = "file not found"
        except Exception as e:
            status[0] = f"unexpected error: {e}"

    status = ["installing"]
    thread = threading.Thread(target=pip_install)
    thread.start()

    # Show progress bar while installation is ongoing
    with tqdm(total=1, bar_format="{l_bar}{bar}| {elapsed}", position=0) as pbar:
        while thread.is_alive():
            time.sleep(0.1)
            pbar.update(0)  # Keep bar active
        pbar.update(1)

    # Final message
    if status[0] == "done":
        print("Requirements installed successfully.")
    elif status[0] == "error":
        print(" An error occurred during installation.")
    elif status[0] == "file not found":
        print(f"Requirements file '{requirements_file}' not found.")
    else:
        print(f"{status[0]}")
#***************************************************************************
def rmsd(model, x_data, y_data, **params):
    '''
     Compute the RMSD for a given model and parameters.
    
     Parameters:
       model: callable
           The model function. It should accept the independent variable (x) as its first argument
           and then any additional parameters as keyword arguments.
       x_data: array-like
          The independent variable data.
       y_data: array-like
          The observed data to compare against.
       **params: dict
           The parameters to pass to the model.
    
     Returns:
       The root mean squared difference.
    '''
    
    # Use list comprehension to compute the model predictions for each x-data point.
    predictions = np.array([model(x, **params) for x in x_data])
    #print(predictions)
    #print(y_data)
    # Compute RMSD.
    rmsd_value = ((np.mean((predictions - y_data) ** 2))**0.5)
    #print(rmsd_value)
    
    return rmsd_value