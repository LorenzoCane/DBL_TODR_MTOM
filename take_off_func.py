import numpy as np
import inspect
from iminuit import Minuit
from iminuit.cost import LeastSquares
from utils import ComplexUnitConverter as conv
from utils import rmsd


def take_off(m, thrust, rho, cl, cd0, k, w_area, airborne_d, margin_coeff=1.15, 
             mu=0., theta=0., lift_frac=1.0, return_velocity=False, dv0 = 1.0, dv_decay = 'exp'):
    vel = 0.0  # m/s
    d = 0.0    # m
    
    theta = conv.convert(theta, 'deg', 'rad')
    cd = cd0 + k * cl**2
    weight = m * 9.81
   
    #print(f'weight = {weight} N')
    while True:
        # Current state
        D = 0.5 * rho * vel**2 * w_area * cd
        
        L = 0.5 * rho * vel**2 * w_area * cl
        #print ('init:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        if L > weight * lift_frac:
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
        L = 0.5 * rho * (vel**2) * w_area * cl
        a_next = (thrust - D - mu * (weight * np.cos(theta) - L) - weight * np.sin(theta)) / m
        #print ('end:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        if  a_current <0.0 : print('stupid')
        a_mean = 0.5 * (a_next + a_current)
        if a_mean <= 0.0:
            break  # Prevent division by zero or deceleration

        v_mean = vel - (0.5 * dv)
        dx = v_mean * dv / a_mean
        d += dx
        #print(dv)
        #print(f'v = {vel}, a = {a_mean}. d = {d}')

    final_distance = (d + airborne_d) * margin_coeff
    return (final_distance, vel) if return_velocity else final_distance

#-----------------------------------------------------------------------------------------------------

def cl_finder(aircraft_mass, to_manuf_value, to_err,thr, rho_isa, 
                cd_0, k_p, wing_area, airborne_dist, safe_margin_coeff, 
                mu, dv0=0.5, dv_decay='const', theta= 0., cl_min=1.0, cl_max=2.0, cl_step=0.01):
    
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
                        return_velocity=False,
                        dv0= 0.5,
                        dv_decay= dv_decay
                    )
    def take_off_wrapper(m, thrust, rho, cl, cd0, k, w_area, airborne_d,
                     margin_coeff=1.15, mu=0., theta=0., lift_frac=1.0, return_velocity=False, dv = dv0):
        results =  np.array([take_off(i, thrust, rho, cl, cd0, k, w_area, airborne_d,
                    margin_coeff=margin_coeff, mu=mu, theta=theta,
                    lift_frac=lift_frac, return_velocity=return_velocity, dv0= dv, dv_decay='const') for i in m])
        return results
    

    # Grid search:
    cl_candidates = np.arange(cl_min, cl_max + cl_step, cl_step)
    #print(cl_candidates)
    rmsd_values = [
        rmsd(take_off, aircraft_mass, to_manuf_value, cl=cl_val, **fixed_params)
        for cl_val in cl_candidates
    ]
    #print(rmsd_values)

    best_cl_guess = cl_candidates[np.argmin(rmsd_values)]
    print(f"Best candidate C_l from grid search: {best_cl_guess:.2f} "
          f"with RMSD = {min(rmsd_values):.2f}")
    

    
    # Create the LeastSquares cost function.
    cost = LeastSquares(aircraft_mass, to_manuf_value, to_err, take_off_wrapper)
    
    # Initialize Minuit. Only cl remains free.
    m = Minuit(cost,
               thrust=thr,
               rho=rho_isa,
               cl=best_cl_guess,
               cd0=cd_0,
               k=k_p,
               w_area=wing_area,
               airborne_d = airborne_dist,
               margin_coeff = safe_margin_coeff,
               mu=mu,
               theta=theta, 
               lift_frac=1.0, 
               return_velocity=False,
               dv = dv0,
               )
    
    # Fix all parameters except cl.
    m.fixed['thrust', 'rho', 'cd0', 'k', 'w_area', 'airborne_d', 'margin_coeff', 'mu', 'theta',
             'lift_frac', 'return_velocity', 'dv'] = True
    m.limits['cl'] = (cl_min, cl_max)
    
    m.migrad()
    m.hesse()
    
    return m.values['cl'], m.errors['cl']

#-----------------------------------------------------------------------------------------------------

def mtom(runway_length, initial_mass, thrust, rho, cl, cd0, k, wing_area, airborne_dist, safety_coef, mu, path_angle):
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

#-----------------------------------------------------------------------------------------------------

def mtom_binary(runway_length, initial_mass, thrust, rho, cl, cd0, k, wing_area, airborne_dist, safety_coef, mu, path_angle, 
                      min_mass=60000, tol=0.5, iter_max = 1.e6):
    low = min_mass
    high = initial_mass
    iter = 0

    while (high - low > tol) :
        mid = (low + high) / 2
        todr = take_off(mid, thrust, rho, cl, cd0, k, wing_area, airborne_dist, margin_coeff=safety_coef, mu=mu, theta=path_angle, return_velocity=False)
        if todr < runway_length:
            low = mid
        else:
            high = mid

        iter +=1

    return low #safety friendly choice