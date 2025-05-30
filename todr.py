import sys 
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv
from tools import rmsd
import numpy as np

def take_off(m, thrust, rho, cl, cd0, k, w_area, airborne_d, margin_coeff=1.15, 
             mu=0.017, theta=0., lift_frac=1.0, v_to=74.5, vel_break = False, return_velocity=False, dv0 = 0.01, dv_decay = 'const'):
    '''
     Simulates the ground roll phase of an aircraft's take-off run and estimates 
     the total take-off distance required (TODR), including the airborne distance.

     Parameters:
     -----------
     m : float
         Aircraft mass in kilograms.
     thrust : float
         Engine thrust in newtons.
     rho : float
         Air density in kg/m^3.
     cl : float
         Lift coefficient (assumed constant).
     cd0 : float
         Zero-lift drag coefficient.
     k : float
         Induced drag factor.
     w_area : float
         Wing area in m^2.
     airborne_d : float
         Airborne distance in meters (after lift-off).
     margin_coeff : float, optional
         Safety margin multiplier applied to the total distance. Default is 1.15.
     mu : float, optional
         Rolling friction coefficient. Default is 0.017.
     theta : float, optional
         Runway slope angle in degrees (positive for uphill). Default is 0°.
     lift_frac : float, optional
         Fraction of weight that must be supported by lift to initiate rotation. Default is 1.0.
     v_to : float, optional
         Target take-off velocity in m/s. Used only if `vel_break=True`. Default is 74.5 m/s.
     vel_break : bool, optional
         If True, terminates ground roll when velocity reaches `v_to`, regardless of lift. Default is False.
     return_velocity : bool, optional
         If True, returns a tuple (take-off distance, lift-off velocity). Default is False.
     dv0 : float, optional
        Base velocity increment (m/s) for simulation time steps. Default is 0.01 m/s.
     dv_decay : str, optional
         Strategy for reducing dv as velocity increases. Options: 
         'const' (constant dv), 'exp' (exponential decay), 
         'exp+' (stronger exponential), 'inv' (inverse). Default is 'const'.

     Returns:
     --------
     float or tuple
         If return_velocity is False: total take-off distance in meters.
         If return_velocity is True: (total take-off distance in meters, lift-off velocity in m/s).
    '''
    vel = 0.0  # m/s
    d = 0.0    # m
    
    theta = conv.convert(theta, 'deg', 'rad')
    cd = cd0 + k * cl* cl
    weight = m * 9.80665
   
    #print(f'weight = {weight} N')
    while True:
        # Current state
        D = 0.5 * rho * vel* vel * w_area * cd #parabolic "classic" drag
        '''
        drag = Drag(ac='A320')
        D = drag.clean(mass=m, tas=vel*1.944, alt=0.0, vs=0.0) #OpenAP drag
        '''
        #D=0.0
        L = 0.5 * rho * vel* vel * w_area * cl
        #print ('init:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        if L >= weight * lift_frac:
            break
        if all([vel_break, vel >= v_to]):
            break

        #dv decay selection
        if dv_decay == 'exp': 
            dv = dv0 * np.exp(-L/weight)
        elif dv_decay == 'exp+':
            dv = dv0 * np.exp(-5.0 * L/weight)           
        elif dv_decay == 'inv':
            dv = dv0 * (weight-L) / (L+1.0) 
        elif dv_decay == 'const':
            dv = dv0
        else:
            raise ValueError(f'dv decay type "{dv_decay}" not supported. Try "exp" or "inv" or "const".\nFor further implemention suggestions please contact developers')

        a_current = (thrust - D - mu * (weight * np.cos(theta) - L) - weight * np.sin(theta)) / m
        
        # Advance velocity
        vel += dv

        # Next state
        D = 0.5 * rho * (vel**2) * w_area * cd
        #D = drag.clean(mass=m, tas=vel*1.944, alt=0.0, vs=0.0) #OpenAP drag
        #D= 0.0
        L = 0.5 * rho * (vel**2) * w_area * cl
        a_next = (thrust - D - mu * (weight * np.cos(theta) - L) - weight * np.sin(theta)) / m
        #print ('end:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        #if  a_current <0.0 : print('stupid')
        a_mean = 0.5 * (a_next + a_current)
        if a_mean <= 0.0:
            break  # Prevent division by zero or deceleration

        v_mean = vel - (0.5 * dv)
        dx = v_mean * dv / a_mean
        d += dx
        #print(dv)
        #print(f'v = {vel}, a = {a_mean}. d = {d}')

    final_distance = (d + airborne_d) * margin_coeff
    #print(f'Model take-off vel: {vel} m/s')
    return (final_distance, vel) if return_velocity else final_distance


def cl_finder(aircraft_mass, to_manuf_value, to_err,thr, rho_isa, 
                cd_0, k_p, wing_area, airborne_dist, safe_margin_coeff,
                mu, dv0=0.01, dv_decay='const', theta= 0., cl_min=1.0, cl_max=2.0, cl_step=0.01):
    '''
     Performs a grid search to estimate the best-fit lift coefficient (C_L)
     that minimizes the RMSD between model-predicted and manufacturer-provided
     take-off distance data for a given aircraft mass.

     The function supports two modes:
     - Original take-off model: Lift-off occurs when L ≥ W × lift_frac
     - Modified take-off model: Lift-off occurs at speed where |L - W| is minimized

     Parameters
     ----------
     aircraft_mass : array-like
         Array of aircraft masses in kg.
     to_manuf_value : array-like
         Manufacturer-provided take-off distances corresponding to aircraft_mass.
     to_err : array-like
         Uncertainty on take-off measurements.
     thr : float
         Thrust in newtons.
     rho_isa : float
         Air density in kg/m^3.
     cd_0 : float
         Zero-lift drag coefficient.
     k_p : float
         Induced drag factor.
     wing_area : float
         Wing area in m^2.
     airborne_dist : float
         Post-liftoff distance (airborne segment) in meters.
     safe_margin_coeff : float
         Safety margin multiplier applied to total distance.
     v_takeoff : float
         Maximum take-off velocity (used to build velocity grid).
     mu : float
         Rolling friction coefficient.
     dv0 : float, optional
         Velocity step size (default: 0.01 m/s).
     dv_decay : str, optional
         Placeholder for decay strategy (not used).
     theta : float, optional
         Runway slope angle in degrees (default: 0°).
     cl_min, cl_max : float
         Minimum and maximum C_L values to test.
     cl_step : float
         Step size for C_L values.
     modified : bool, optional
         Whether to use the modified take-off model (default: False).

     Returns
     -------
     best_cl_guess : float
         Lift coefficient value yielding the lowest RMSD.
     dummy_error : float
         Placeholder for uncertainty (currently set to 0.1).
    '''
    
    fixed_params = dict(thrust=thr,
                        rho=rho_isa,
                        cd0=cd_0,
                        k=k_p,
                        w_area=wing_area,
                        airborne_d = airborne_dist,
                        margin_coeff = safe_margin_coeff,
                        mu=mu,
                        theta=theta,
                        lift_frac=1.0, 
                        vel_break = False,
                        dv0= 0.01,
                        dv_decay= dv_decay
                    )
    
    # Grid search:
    cl_candidates = np.arange(cl_min, cl_max + cl_step, cl_step)

    rmsd_values = [
            rmsd(take_off, aircraft_mass, to_manuf_value, cl=cl_val, **fixed_params)
            for cl_val in cl_candidates
            ]
    #print(rmsd_values)

    best_cl_guess = cl_candidates[np.argmin(rmsd_values)]
    print(f"Best candidate C_l from grid search: {best_cl_guess:.3f} "
          f"with RMSD = {min(rmsd_values):.2f}")
    
    return best_cl_guess, 0.1



def mtom(runway_length, initial_mass, thrust, rho, cl, cd0, k, wing_area, airborne_dist, safety_coef, mu, path_angle):
    '''
     Estimates the Maximum Take-Off Mass (MTOM) such that the total take-off
     distance required (TODR) does not exceed the given runway length.
 
     The function uses a coarse-to-fine grid search strategy by progressively
     decreasing the aircraft mass until TODR fits within the runway, then refining
     the estimate with smaller mass steps.
 
     Parameters
     ----------
     runway_length : float
         Maximum available take-off distance (meters).
     initial_mass : float
         Starting guess for aircraft mass (kg).
     thrust : float
         Engine thrust (N).
     rho : float
         Air density (kg/m^3).
     cl : float
         Lift coefficient.
     cd0 : float
         Zero-lift drag coefficient.
     k : float
         Induced drag coefficient.
     wing_area : float
         Wing surface area (m²).
     airborne_dist : float
         Distance required after lift-off (m).
     safety_coef : float
         Safety margin applied to TODR (e.g., 1.15).
     mu : float
         Friction coefficient during ground roll.
     path_angle : float
         Runway slope angle (degrees).
 
     Returns
     -------
     mass : float
         Estimated MTOM (kg) that ensures TODR ≤ runway_length.
    '''

    mass = initial_mass
    iter = 0
    if take_off(mass, thrust, rho, cl, cd0, k, wing_area, airborne_dist,margin_coeff=safety_coef, mu=mu, theta=path_angle, return_velocity=False) < runway_length:
        return mass  # already feasible

    for step in [1e3, 1e2, 1e1, 1]:  # reduce mass by 1000, 100, 10, 1
        while True:
            todr = take_off(mass, thrust, rho, cl, cd0, k, wing_area, airborne_dist, safety_coef, mu, path_angle)
            if todr < runway_length:
                mass += step  # step back up to refine
                break
            mass -= step  # keep reducing
            iter +=1
            #print(f'Iter #{iter+1}: MTOM = {mass}, TODR = {todr}, l_runway = {runway_length}')

    return mass