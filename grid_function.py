import numpy as np
import os
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import rotate
from pyproj import Proj
from utils import ComplexUnitConverter as conv
from utils import rmsd

#-----------------------------------------------------------------------------------------------------
def AOC_change(phi1, rho1, rho2, unit = 'deg'):
    '''
    '''
    phi1_rad = conv.convert(phi1, 'deg', 'rad')
    phi2_rad = np.arcsin( np.sin(phi1_rad) * rho2 / rho1 ) # correct if trhust and drag are considered proportional to air density

    return phi2_rad if unit == 'rad' else conv.convert(phi2_rad, 'rad', 'deg') #possible to have different unit as output

#-----------------------------------------------------------------------------------------------------
def noise_calc(x_dist, y_dist, z_dist, sound_level, alpha = 1.0e-4, eps = 1.0e-6):
    '''
    '''
    #calculate aircraft-point distance (squared)
    r_sq = (x_dist*x_dist + y_dist*y_dist + z_dist*z_dist)

    #sound absorb. coeff???
    #Sound level in a point at distance r 
    L_db = sound_level - 10.0 * np.log10(2.0 * np.pi * r_sq + eps) #dB

    return L_db
#-----------------------------------------------------------------------------------------------------
def grid_def(scale, runway_lenght, grid_points = 200, center_offset=(0.0, 0.0)):
    '''
     Define a 2D ground grid centered on the runway midpoint.

     Parameters
     ----------
     runway_length : float
         Length of the runway in meters.
     grid_scale : float, optional
         How many runway‐lengths to extend each direction (default = 10).
     grid_res : int, optional
         Number of points along each axis (default = 200).
     center_offset : tuple of floats, optional
         (dx, dy) shift to apply to the grid center (default = (0,0)).

     Returns
     -------
     X, Y : 2D np.ndarray
         Meshgrid arrays of shape (grid_res, grid_res).
     x, y : 1D np.ndarray
         The coordinate vectors for each axis.
    '''
    #create grid
    grid_size =  scale * runway_lenght #size of the grid
    x =  np.linspace(-0.5 * grid_size, 0.5 * grid_size, grid_points) + center_offset[0]
    y = np.linspace(-0.5 * grid_size, 0.5 * grid_size, grid_points)  + center_offset[1]
    #create mesh
    X,Y =  np.meshgrid(x,y)

    return X,Y,x, y, grid_size
#-----------------------------------------------------------------------------------------------------
def trajectory_s2n(runway_lenght, grid_size, TODR, climb_angle_deg, extra_frac = 0.25, 
                   n_point = 200, point_ratio= 0.3):
    '''
    '''
    #calculte the how far (horiz) we need to go
    target_y  = (0.5 + extra_frac) * grid_size
    #divide evaluation point
    n_roll = round(n_point * point_ratio)
    n_climb =  n_point - n_roll

    #roll phase
    y0 = - 0.5 * runway_lenght
    roll_y = np.linspace(y0, y0 + TODR, n_roll)
    roll_x = np.zeros_like(roll_y) #south to north
    roll_z = np.zeros_like(roll_y) #still on the ground

    #climb phase
    climb_dist = target_y - y0 - TODR #distance to be covered in climb phase
    climb_p = np.linspace(0, climb_dist, n_climb)
    climb_y = y0 + TODR + climb_p    
    climb_x = np.zeros_like(climb_y)
    climb_z = climb_p * np.sin(conv.convert(climb_angle_deg, 'deg', 'rad'))

    #concatenate phases
    return (np.concatenate([roll_x, climb_x]),
            np.concatenate([roll_y, climb_y]),
            np.concatenate([roll_z, climb_z]))   
#-----------------------------------------------------------------------------------------------------
def noise_grid(runway_length, TODR, climb_angle_deg, sound_level, alpha = 1.0e-4, grid_scale=10, grid_points=200, 
               extra_frac = 0.25, npoints = 300, eps = 1.0e-6):
    '''
    '''

    #create grid
    X,Y,x, y, grid_size = grid_def(runway_length, grid_scale, grid_points)  

    #build trajectory
    air_x, air_y , air_z = trajectory_s2n(runway_length, grid_size, TODR, climb_angle_deg, 
                                          extra_frac=extra_frac, n_point=npoints)  
    
    #compute noise
    range_len =  len(air_x)
    Lp_ts = np.zeros((range_len, grid_points, grid_points))

    for t, (ax, ay, az) in enumerate(zip(air_x, air_y, air_z)):
        dx = X - ax
        dy = Y - ay
        dz = az
        Lp_ts[t] = noise_calc(dx, dy, dz, sound_level, alpha=alpha, eps=eps)

    Lp_max  = Lp_ts.max(axis=0)
    Lp_mean = Lp_ts.mean(axis=0) 
    
    return  X, Y, Lp_ts, Lp_max, Lp_mean, air_x, air_y, air_z

#-----------------------------------------------------------------------------------------------------
def rotate_grid(X, Y, angle_deg):
    """
    Rotate grid clockwise to align with real-world runway heading.
    Use reshape=True to avoid clipping.
    """
    # Rotate the noise layer
    #Lp_rotated = rotate(Lp_max, angle=angle_deg, reshape=False, order=1)  # bilinear
    X_rot, Y_rot = rotate_grid(X, Y, angle_deg)
    return X_rot, Y_rot #, Lp_rotated
#-----------------------------------------------------------------------------------------------------
def project_to_latlon(X, Y, lat0, lon0):
    proj_laea = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0)
    lon_grid, lat_grid = proj_laea(X, Y, inverse=True)
    return lat_grid, lon_grid

#-----------------------------------------------------------------------------------------------------
def plot_db_contours(lat_grid, lon_grid, Lp_max, output_name, output_path, levels=[50, 55, 60]):

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_title('Aircraft Noise Contours (dB)')
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.gridlines(draw_labels=True)

    cs = ax.contour(lon_grid, lat_grid, Lp_max, levels=levels, colors=['blue', 'green', 'red'], linewidths=2)
    ax.clabel(cs, fmt='%d dB', fontsize=10)

    plt.savefig(os.path.join(output_path, output_name))
