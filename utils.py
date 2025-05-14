import numpy as np



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