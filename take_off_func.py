import numpy as np
import inspect
from iminuit import Minuit
from iminuit.cost import LeastSquares
from utils import ComplexUnitConverter as conv
from utils import rmsd
from openap.drag import Drag


def take_off(m, thrust, rho, cl, cd0, k, w_area, airborne_d, margin_coeff=1.15, 
             mu=0.017, theta=0., lift_frac=1.0, v_to=74.5, vel_break = False, return_velocity=False, dv0 = 0.01, dv_decay = 'const'):
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

#-----------------------------------------------------------------------------------------------------
def take_off_modified(m, thrust, rho, cl, cd0, k, w_area, airborne_d, margin_coeff=1.15, 
             mu=0.02, theta=0., lift_frac=1.0, v_to=150.0, vel_break = False, return_velocity=False, dv0 = 0.01, dv_decay = 'const'):
    vel = 0.0  # m/s
    d = 0.0    # m
    
    theta = conv.convert(theta, 'deg', 'rad')

    cd = cd0 + k * cl* cl
    weight = m * 9.81
    speeds = np.arange(0, v_to, dv0)
    
    L_arr = 0.5 * rho * speeds**2 * w_area * cl
    D_arr = 0.5 * rho * speeds**2 * w_area * cd
    a_arr = (thrust - D_arr - mu * (weight * np.cos(theta) - L_arr) - weight * np.sin(theta)) / m

    # Identify index where |L - W| is minimized
    lift_diff = np.abs(L_arr - weight)
    idx_takeoff = np.argmin(lift_diff)

    # Compute distance using trapezoidal method
    dx_arr = []
    for i in np.arange(1, len(speeds[0 : (idx_takeoff + 2)])):
        v_avg = (speeds[i] + speeds[i - 1]) / 2
        a_avg = (a_arr[i] + a_arr[i - 1]) / 2
        if a_avg <= 0:
            break
        dx = v_avg * dv0 / a_avg
        dx_arr.append(dx)

    ground_roll = np.sum(dx_arr, initial=0.0)
    airborne_distance = airborne_d
    final_todr = (ground_roll + airborne_distance) * margin_coeff
    final_velocity = speeds[idx_takeoff]

    return (final_todr, final_velocity) if return_velocity else final_todr

#-----------------------------------------------------------------------------------------------------

def cl_finder(aircraft_mass, to_manuf_value, to_err,thr, rho_isa, 
                cd_0, k_p, wing_area, airborne_dist, safe_margin_coeff, v_takeoff, 
                mu, dv0=0.01, dv_decay='const', theta= 0., cl_min=1.0, cl_max=2.0, cl_step=0.01, modified =False):
    
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
                        v_to = v_takeoff,
                        dv_decay= dv_decay
                    )
    if modified :
        def take_off_wrapper(m, thrust, rho, cl, cd0, k, w_area, airborne_d,
                     margin_coeff=1.15, mu=0.017, theta=0., lift_frac=1.0, return_velocity=False, dv = dv0):
            results =  np.array([take_off_modified(i, thrust, rho, cl, cd0, k, w_area, airborne_d, v_to= v_takeoff,
                    margin_coeff=margin_coeff, mu=mu, theta=theta,
                    lift_frac=lift_frac, return_velocity=return_velocity, vel_break = False, dv0= dv, dv_decay='const') for i in m])
            return results
    else:
        def take_off_wrapper(m, thrust, rho, cl, cd0, k, w_area, airborne_d,
                     margin_coeff=1.15, mu=0.017, theta=0., lift_frac=1.0, return_velocity=False, dv = dv0):
            results =  np.array([take_off(i, thrust, rho, cl, cd0, k, w_area, airborne_d, v_to= v_takeoff,
                    margin_coeff=margin_coeff, mu=0.02, theta=theta,
                    lift_frac=lift_frac, return_velocity=return_velocity, vel_break = False, dv0= dv, dv_decay='const') for i in m])
            return results
    

    # Grid search:

    cl_candidates = np.arange(cl_min, cl_max + cl_step, cl_step)
    #print(cl_candidates)
    if modified:
        rmsd_values = [
            rmsd(take_off_modified, aircraft_mass, to_manuf_value, cl=cl_val, return_velocity = False, **fixed_params)
            for cl_val in cl_candidates
            ]
    else:    
        rmsd_values = [
            rmsd(take_off, aircraft_mass, to_manuf_value, cl=cl_val, **fixed_params)
            for cl_val in cl_candidates
            ]
    #print(rmsd_values)

    best_cl_guess = cl_candidates[np.argmin(rmsd_values)]
    print(f"Best candidate C_l from grid search: {best_cl_guess:.3f} "
          f"with RMSD = {min(rmsd_values):.2f}")
    

    '''
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
    '''
    return best_cl_guess, 0.1
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