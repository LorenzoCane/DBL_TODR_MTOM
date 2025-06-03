# Aircraft Performance Evaluation

## Overview 
This model estimates the performance of a given aircraft at a specific airport under historical scenario and various future climate change scenarios. It computes: 
 - **Take-Off Distance Required (TODR)**

 - **Maximum Take-Off Mass (MTOM)**

 - **Mass restrictions due to runway length limitations**
 

 [ISA conditions](https://aerodynamics4students.com/properties-of-the-atmosphere/), to calibrate model parameters, such as the [Lift coefficient](https://en.wikipedia.org/wiki/Lift_coefficient), $C_L$.

It also leverages pre-processed climate data and projections from the [Coupled Model Intercomparison Project Phase 6 (CMIP6)](https://pcmdi.llnl.gov/CMIP6/) models.


## Repository structure

- `main.py`: Main script to execute the full pipeline: parameter evaluation, data processing, and result visualization.

- `atm_data_preprocessor.py`: Preprocesses raw `.nc` climate data and creates `.csv`files based on selected years, months, and atmospheric variables.

- `atm_anal_func.py`: Functions for atmospheric calculations used in `atm_data_preprocessor.py` .

- `todr.py` : Functions for calculating TODR, MTOM, and mass restrictions, including $C_L$ optimization.

- `performance_evaluation.py`:  Evaluates aircraft performance and saves results as `.parquet` file.

- `plots.py`: Generates plots from processed results.

- `config.yml`: Configuration file containing parameters for data processing and modeling.

- `requirements.txt`: List of Python dependencies required to run the project.

- `aircraft_util.py`. Utilities to retrieve aircraft characteristics from manufacturer data and the [OpenAP](https://openap.dev/openap.html) library.

- `airport_utils.py`:  Utilities to retrieve airport data from internal runway datasets and the  [airportsdata](https://pypi.org/project/airportsdata/) package.

- `cl_calc.py`: Estimates the lift coefficient from manufacturer data and generates the corresponding  `.parquet` file. 

- `config_loader.py` : Function used to load the configuration file.

- `constants.py`: Contains physical and model constants used across the pipeline.

- `utils/` : Utility functions for unit conversions, grid management, file handling, and more.


## Getting Started

Ensure you have **Python 3.8 or higher** installed. Alternatively, you can request a **Google Colab notebook version** if preferred.
To run the project locally, place the required data files in the following directories:
 - Raw climate data and runways lenght dataset $→$ `./data/raw`.
 - Manufacturer take-off performance data $→$ `./cl_data/TODR_MTOM_manuf`. 
> :warning: **PLEASE BE AWARE OF THE NAMING CONVENTIONS USED**.

### Running the entire pipeline
To ensure all necessary `.parquet` files and directories are created, it is recommended to run the entire pipeline on first execution:
```
python main.py
```
This will:
   1. Fit the lift coefficient,

   2. Process climate and aircraft data,

   3. Evaluate performance,

   4. Generate plots and output files.

### Running specific part(s)
After initial setup, you can run individual components by:

   - Modifying the **CAP** control variables at the beginning of `main.py`, or

   - Running each script independently (e.g., `cl_calc.py`, `performance_evaluation.py`).
            