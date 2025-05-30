import numpy as np

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
    # Compute RMSD.
    rmsd_value = ((np.mean((predictions - y_data) ** 2))**0.5)
    
    return rmsd_value