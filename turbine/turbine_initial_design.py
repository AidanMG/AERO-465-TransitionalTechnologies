import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import interpolate as intp
from scipy.optimize import fsolve, least_squares


# Conversion constants
in_to_mm = 25.4
ft_to_m = in_to_mm*12/1000
Cp_g = 1148 #J/kgK
Cp_air = 1005
R_air = 287

# Container class for complete velocity triangle information for all stages
@dataclass
class VelocityTriangle:
    U : float
    V1 : float
    V2 : float
    V3 : float
    alpha1 : float
    alpha2 : float
    alpha3 : float
    alpha_r2 : float
    alpha_r3 : float

@dataclass
class Pressure:
    P01: float
    P02: float
    P03: float
    P1: float
    P2: float
    P3: float

@dataclass
class Temperature:
    T01: float
    T02: float
    T03: float
    T1: float
    T2: float
    T3: float    


# GIVEN CONSTRAINTS THAT WILL NOT CHANGE (see course outline)
# Stage
class Stage:
    M_in = 0.15
    swirl_in = np.deg2rad(-10) # swirl -10 degrees relative to axial
    M_out_min, M_out_max = 0.30, 0.60
    swirl_out_min, swirl_out_max = np.deg2rad(5),np.deg2rad(40)

# Vane
class Vane:
    AR = 0.5
    zw_Min, zw_Max = 0.75, 0.90
    trailing_t = 0.05*in_to_mm # minimum

# Blade
class Blade:
    AR = 1.3
    zw_Min, zw_Max = 0.80, 1.00
    trailing_t = 0.03*in_to_mm # minimum
    tip_clearance_min, tip_clearance_max = 0.01, 0.02 # Ratio
    AN2_max = 4.5E10 * (in_to_mm/1000)**2 #(m**2)*(RPM**2)
    rim_speed_max = 1000 * ft_to_m #m/s
    
# Boundary conditions derived in Part A, Standard SI units unless noted otherwise 
class Conditions:
    T0_pre = 515.992 # Temperature before combustion, used in RDTF computations
    T01 = 1132.865 # Turbine inlet temp, average
    T02 = 1132.865 # no work, adiabatic
    T02_mix = 1112.203 # After mixing with bleed air
    T03 = 910.521 # Turbine outlet temperature before mixing
    w = (T02_mix-T03)*Cp_g
    
    mdot_5 = 6.396 # mdot at the intake
    mdot_5i = 6.729 # mdot going across the turbine blades, in kg/s
    # w = W_dot/mdot_5i

    RDTF = 0.05
    T01_max = (1+RDTF)*T01 - RDTF*T0_pre # Based on RDTF = (Tmax-Tavg)/(Tavg-Tpre)

    # Assuming a parabolic radial temperature profile peaking at 50% span (y/b=0.5):
    #       T01(y/b) = A*(y/b - 0.5)**2 + T01_max
    # Since Integral(0->1) (T01)dy === Tavg, performing the integration yields the constant A=T01_slope
    T01_slope = (-3/2) * (T01_max-T01)/(0.5**3)
    
    
    P01 = 592.845 # kPa
    P03 = 215.888 # kPa
    
    T01_var = lambda y : Conditions.T01_slope * (y - 0.5)**2 + Conditions.T01_max


def temperature_ratio(M, gamma=1.333):
    """T0/T
    """
    return 1+(gamma-1)/2*M**2

def pressure_ratio(M, gamma=1.333):
    """P0/P
    """ 
    return temperature_ratio(M, gamma)**((gamma)/(gamma-1))

def sound(T, R=287, gamma=1.333):
    """Speed of sound (m/s)
    """
    return np.sqrt(T*R*gamma)

# Steps:
# Using initial design values and requirements (Turbine inlet temp, velocity, pressure, Turbine power requirement)
    # Determine best velocity triangles from flow properties
    # Determine geometric parameters (No. vanes, blades, metal angle, stagger angle, gaspath)
# Procedure A
# Establish midspan velocity triangles
# Assume T-based reaction, get RPM, radii from structural reqs
# Free-vortex theory to determine velocity triangles at hub and tip
# Determine geometric parameters (stagger angle, throat opening, nominal incidence, deviation, etc.)
# from loss correlations, provide individual losses and overall loss, impact on turbine efficiency

# Procedure B
# Assumne radii
# Assume a reaction and RPM
# Establish the velocity triangles
# Iterate above to satisfy requirements with given structural constraints
# Continue rest as above

# Using procedure B
N = 100
def get_blade_from_RPM(n, rim_speed=Blade.rim_speed_max, AN2=Blade.AN2_max):
    """Gets the blade basic geometry parameters from the RPM

    Args:
        n (float): RPM
    
    Returns:
        blade_rh (float): hub radius, meters
        blade_rt (float): tip radius, meters
    """
    blade_rh = rim_speed/(n*np.pi/30) # Blade hub radius
    blade_rt = np.sqrt(AN2/(np.pi*n**2)+blade_rh**2) # Blade tip radius
    return blade_rh, blade_rt

def plot_blades(rim_speed=Blade.rim_speed_max, AN2 = Blade.AN2_max):
    # Plotting Blade RPM effect on blade radius
    blade_RPM_range = np.linspace(10000,25000,N) #RPM from 10k to 25k

    blade_rh_range, blade_rt_range = get_blade_from_RPM(blade_RPM_range, rim_speed, AN2)

    blade_length_range = (blade_rt_range - blade_rh_range) # Total blade radial length
    blade_rm_range = blade_length_range/2  + blade_rh_range # Blade midspan radius


    fig, ax = plt.subplots()
    ax.plot(blade_RPM_range, blade_rt_range, 'b')
    ax.plot(blade_RPM_range, blade_rm_range, 'r')
    ax.plot(blade_RPM_range, blade_rh_range, 'b')
    ax.fill_between(blade_RPM_range, blade_rh_range, blade_rt_range, alpha=0.2,)
    ax.set_xlim(blade_RPM_range[0], blade_RPM_range[-1])
    ax.set_ylim(0,blade_rt_range[0]*1.1)
    ax.set_xlabel("RPM")
    ax.set_ylabel("Blade Dimensions (m)")


def get_midspan_velocity_triangles(flow_coeff, reaction, area_ratio):
    """Get velocity triangles at the midspan from flow coefficient, reaction, and area ratio
    NO LONGER USED -- Assumes constant Ca/density which is not at all accurate

    Args:
        flow_coeff (_type_): _description_
        reaction (_type_): _description_
        area_ratio (_type_): _description_

    Returns:
        _type_: _description_
    """

    ### Conditions at the inlet ###
    T1 = Conditions.T01/temperature_ratio(Stage.M_in) # Temperature at the first stage
    V1 = Stage.M_in*sound(T1) # Velocity at the first stage
    alpha1 = Stage.swirl_in


    Va = V1*np.cos(alpha1) # Axial velocity of first stage
    Va2 = Va*area_ratio


    # Renaming all the variables to be more similar to our formula sheets....
    # Assuming Va = cst for initial design
    U = Va2 / flow_coeff
    phi = flow_coeff
    R = reaction
    W = Conditions.w

    ##### FINDING THE ABSOLUTE FLOW ANGLES #####
    # From the equations :
    # R = 1 + 0.5*phi*(tan(alpha3) - tan(alpha2))
    #      => tan(alpha3) - tan(alpha2) = (R-1)/(0.5phi)
    # W = UVa (tan(alpha3) + tan(alpha2))
    #      => tan(alpha3) + tan(alpha2) = W/(U Va)
    # Therefore:
    #       alpha3 = atan(0.5*(W/(U Va) + (R-1)/(0.5phi)))
    #       alpha2 = atan(0.5*(W/(U Va) - (R-1)/(0.5phi)))

    alpha2 = np.arctan(0.5*(W/(U*Va2) - (R-1)/(0.5*phi)))
    alpha3 = np.arctan(0.5*(W/(U*Va2) + (R-1)/(0.5*phi)))

    ##### FINDING THE RELATIVE FLOW ANGLES #####
    # From the formulas
    # (1/phi) = tan(alpha_r3)-tan(alpha3) = tan(alpha2) - tan(alpha_r2) 

    alpha_r2 = np.arctan(np.tan(alpha2) - 1/phi)    
    alpha_r3 = np.arctan(np.tan(alpha3) + 1/phi)

    ##### FINDING THE STAGE VELOCITIES #####
    V2 = Va2 / np.cos(alpha2)
    V3 = Va2 / np.cos(alpha3)

    # Cast V1 to the same dimension as flow_coeff for easier plotting
    V1 = np.full_like(flow_coeff, V1)
    return_val = VelocityTriangle(U,V1,V2,V3,alpha1,alpha2,alpha3,alpha_r2,alpha_r3)

    return return_val

def analyze_triangle_range(flow_coeffs, reactions, area_ratio):
    flow_grid, reaction_grid = np.meshgrid(flow_coeffs, reactions)
    vel_tri = get_midspan_velocity_triangles(flow_grid, reaction_grid, area_ratio)

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(16,9)    
    fig.suptitle(f"Ratio: {area_ratio}")
    plt.set_cmap('jet')


    # Filter out values outside of the min and max ranges
    vel_tri.alpha3 = np.where((vel_tri.alpha3 >= Stage.swirl_out_min) & (vel_tri.alpha3 <= Stage.swirl_out_max), vel_tri.alpha3, np.nan) # Swirl Angle Range
    vel_tri.V3 = np.where((vel_tri.V3 >= Stage.M_out_min*sound(Conditions.T03/temperature_ratio(Stage.M_out_min))) & (vel_tri.V3 <= Stage.M_out_max*sound(Conditions.T03/temperature_ratio(Stage.M_out_max))), vel_tri.V3, np.nan) # Outlet Mach Range
    vel_tri.V2 = np.where((vel_tri.V2 <= sound(Conditions.T02/temperature_ratio(1))), vel_tri.V2, np.nan)

    # Apply mask for all conditions to alpha3
    vel_tri.alpha3 = np.where((np.isnan(vel_tri.V3) | (np.isnan(vel_tri.V2))), np.nan, vel_tri.alpha3)
    
    # Apply mask for alpha3 to the other conditions
    vel_tri.V3 = np.where(np.isnan(vel_tri.alpha3), np.nan, vel_tri.V3)
    vel_tri.V2 = np.where(np.isnan(vel_tri.alpha3), np.nan, vel_tri.V2)
    vel_tri.U = np.where(np.isnan(vel_tri.alpha3), np.nan, vel_tri.U)

    N = 1000
    u_contour = axs[0,0].contourf(flow_grid, reaction_grid, vel_tri.U, levels=N)
    u_cbar = plt.colorbar(u_contour, ax=axs[0,0])
    u_cbar.set_label("U (m/s)")



    v2_contour = axs[1,0].contourf(flow_grid, reaction_grid, vel_tri.V2, levels=N)
    v2_cbar = plt.colorbar(v2_contour, ax=axs[1,0])
    v2_cbar.set_label("V2 (m/s)")



    v3_contour = axs[0,1].contourf(flow_grid, reaction_grid, vel_tri.V3, levels=N)
    v3_cbar = plt.colorbar(v3_contour, ax=axs[0,1])
    v3_cbar.set_label("V3 (m/s)")




    alpha3_contour = axs[1,1].contourf(flow_grid, reaction_grid, np.rad2deg(vel_tri.alpha3), levels=N)
    alpha3_cbar = plt.colorbar(alpha3_contour, ax=axs[1,1])
    alpha3_cbar.set_label("alpha3 (deg)")

    for ax in axs.flatten():
        ax.set_xlabel("phi")
        ax.set_ylabel("R")

def get_reaction(vel_tri: VelocityTriangle):
    """Get reaction degree from velocity triangle
    NO LOGNER USED - Assumes constant density
    """

    phi = vel_tri.V2*np.cos(vel_tri.alpha2)/vel_tri.U
    return 0.5*phi*(np.tan(vel_tri.alpha_r3)-np.tan(vel_tri.alpha_r2))
    
def get_temp_reaction(T01s, T02s, T03s, V1s, V2s, V3s):
    """Get temperature-based reaction from speeds and total temps at locations

    Args:
        T01s (_type_): _description_
        T02s (_type_): _description_
        T03s (_type_): _description_
        V1s (_type_): _description_
        V2s (_type_): _description_
        V3s (_type_): _description_

    Returns:
        _type_: _description_
    """
    T1s = T01s - (V1s)**2/(2*Cp_g)
    T2s = T02s - (V2s)**2/(2*Cp_g)
    T3s = T03s - (V3s)**2/(2*Cp_g)
    return (T2s - T3s)/ (T1s - T3s)


def get_mach_offset(T01s, T02s, T03s, V1s, V2s, V3s):
    """Get mach number offset at a certain location -- feed velocities and total temps to get the Mach

    Args:
        T01s (_type_): _description_
        T02s (_type_): _description_
        T03s (_type_): _description_
        V1s (_type_): _description_
        V2s (_type_): _description_
        V3s (_type_): _description_

    Returns:
        _type_: _description_
    """
    T1s = T01s - (V1s)**2/(2*Cp_g)
    T2s = T02s - (V2s)**2/(2*Cp_g)
    T3s = T03s - (V3s)**2/(2*Cp_g)

    M1s = V1s / sound(T1s)
    M2s = V2s / sound(T2s)
    M3s = V3s / sound(T3s)
    return M1s, M2s, M3s    

def get_mach_offset_2(T1s, T2s, T3s, V1s, V2s, V3s):
    """Gets mach number offset if static temperatures are defined instead

    Args:
        T1s (_type_): _description_
        T2s (_type_): _description_
        T3s (_type_): _description_
        V1s (_type_): _description_
        V2s (_type_): _description_
        V3s (_type_): _description_

    Returns:
        _type_: _description_
    """
    M1s = V1s / sound(T1s)
    M2s = V2s / sound(T2s)
    M3s = V3s / sound(T3s)
    return M1s, M2s, M3s        

# def get_offset_velocity_dict(d: dict, r0, r_offset):
#     dict2 = d.copy()
#     ratio = r0/r_offset
#     U = dict2["U"]/ratio
    
#     Va1 = np.cos(dict2[""].alpha2)*vel_tri.V2
#     Va2 = np.cos(dict2[""].alpha2)*vel_tri.V2
#     Va3 = np.cos(dict2[""].alpha3)*vel_tri.V3 
#     alpha2 = np.arctan(ratio*np.tan(vel_tri.alpha2))
#     alpha3 = np.arctan(ratio*np.tan(vel_tri.alpha3))
    
#     V2 = Va2/np.cos(alpha2)
#     V3 = Va3/np.cos(alpha3)
   
#     alpha_r2 = np.arctan(np.tan(alpha2) - (U/Va2))
#     alpha_r3 = np.arctan(np.tan(alpha3) + (U/Va3))
    
#     return VelocityTriangle(U, vel_tri.V1, V2, V3, vel_tri.alpha1, alpha2, alpha3, alpha_r2, alpha_r3)    

    return

def get_offset_triangle(vel_tri: VelocityTriangle, r0, r_offset):
    """Get velocity triangle at offset position based on the midspan velocity triangle, using free vortex.
    NOTE: This procedure assumes constant radius throughout the section.

    Args:
        vel_tri (_type_): _description_
        r0 (_type_): _description_
        r_offset (_type_): _description_
    """
    ratio = r0/r_offset
    U = vel_tri.U/ratio
    
    Va1 = np.cos(vel_tri.alpha2)*vel_tri.V2
    Va2 = np.cos(vel_tri.alpha2)*vel_tri.V2
    Va3 = np.cos(vel_tri.alpha3)*vel_tri.V3 
    alpha2 = np.arctan(ratio*np.tan(vel_tri.alpha2))
    alpha3 = np.arctan(ratio*np.tan(vel_tri.alpha3))
    
    V2 = Va2/np.cos(alpha2)
    V3 = Va3/np.cos(alpha3)
   
    alpha_r2 = np.arctan(np.tan(alpha2) - (U/Va2))
    alpha_r3 = np.arctan(np.tan(alpha3) + (U/Va3))
    
    return VelocityTriangle(U, vel_tri.V1, V2, V3, vel_tri.alpha1, alpha2, alpha3, alpha_r2, alpha_r3)


def plot_triangle_reaction_ranges(triangle_midspan, RPM, rim_speeds, an2s):
    """Plot ranges of offset velocity triangles

    Args:
        triangle_midspan (_type_): _description_
        RPM (_type_): _description_
        rim_speeds (_type_): _description_
        an2s (_type_): _description_
    """
    num = 100

    rim_grid, an2_grid = np.meshgrid(rim_speeds, an2s)
    
    rh_grid, rt_grid = get_blade_from_RPM(RPM, rim_grid, an2_grid)
    rm_grid = (rh_grid + rt_grid) / 2
    
    triangle_hub = get_offset_triangle(triangle_midspan, rm_grid, rh_grid)
    triangle_tip = get_offset_triangle(triangle_midspan, rm_grid, rt_grid)
    
    hub_reaction = get_reaction(triangle_hub)
    # hub_reaction = np.where(((hub_reaction > 0) & (hub_reaction < 1)), hub_reaction, np.nan)
    
    tip_reaction = get_reaction(triangle_tip)
    # tip_reaction = np.where(((tip_reaction > 0) & (tip_reaction < 1) ), tip_reaction, np.nan)
    
    
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(16,9)
    hub_contour = axs[0].contourf(100*rim_grid/Blade.rim_speed_max, 100*an2_grid/Blade.AN2_max, hub_reaction, levels=num)
    hub_cbar = plt.colorbar(hub_contour, ax=axs[0])
    hub_cbar.set_label("Hub Reaction")
    
    tip_contour = axs[1].contourf(100*rim_grid/Blade.rim_speed_max, 100*an2_grid/Blade.AN2_max, tip_reaction, levels=num)
    tip_cbar = plt.colorbar(tip_contour, ax=axs[1])
    tip_cbar.set_label("Tip Reaction")
    
    for ax in axs:
        ax.set_xlabel("Rim speed, % of structural limit")
        ax.set_ylabel("AN2, % of structural limit")

def full_solver(M1s, alpha1s, ARs, alpha2s, Yps, T01s):
    """NON WORKING code that cuts out a few variables from the used method by instead assuming you know Yps"""
    # Reshape all arrays to 1D for solver
    shape = M1s.shape
    M1s = M1s.ravel()
    alpha1s = alpha1s.ravel()
    ARs = ARs.ravel()
    alpha2s = alpha2s.ravel()
    Yps = Yps.ravel()
    
    
    def func(M2):
        return (M1s/M2)*(ARs)*(np.cos(alpha1s)/np.cos(alpha2s))*(F1(M1s)/F2(M2, Yps)) - 1
    
    def F1(M1):
        """(1+(gamma-1)/2 M**2)^(1/2 - gamma/(gamma-1))
        """
        return temperature_ratio(M1)**(0.5-(1.333/0.333))
    
    def F2(M2, Yp):
        """F1 but accounting for pressure loss Yp from 1 to 2
        """
        return F1(M2)/(1-Yp/pressure_ratio(M2))
    
    # Array of Mach Numbers, reshape back into 2d array
    M2s = fsolve(func, M1s)
    M2s = M2s.reshape(shape)
    Yps = Yps.reshape(shape)
    
    # Use mach number to solve for T, V, P0, P
    ys = np.linspace(0,1,len(M1s),endpoint=True)
    
    T02s = T01s
    T2s = T02s/temperature_ratio(M2s)
    V2s = M2s*sound(T2s)
    P2s = F2(M2s, Yps)*Conditions.P01
    P02s = P2s*pressure_ratio(M2s)
    
    return M2s, V2s, T2s, T02s, P2s, P02s

def condition_plotter(M2s, V2s, T2s, T02s, P2s, P02s, title=None):
    """Baseline code to plot the conditions along the blade
    
    """
    fig, axs = plt.subplots(3,2)
    plt.subplots_adjust(hspace=0.5)
    if title:
        plt.suptitle(title)
    
    xs = np.linspace(0,1,len(M2s),endpoint=True)
    xs, ys = np.meshgrid(xs, xs)
    fig.set_size_inches(9,16)
    
    M2plot = axs[0][0].contourf(xs, ys, M2s, levels=100)
    fig.colorbar(M2plot, ax=axs[0][0])
    axs[0][0].set_title("M")
    V2plot = axs[0][1].contourf(xs, ys, V2s, levels=100)
    fig.colorbar(V2plot, ax=axs[0][1])
    axs[0][1].set_title("V (m/s)")
    T02plot = axs[1][0].contourf(xs, ys, T02s, levels=100)
    fig.colorbar(T02plot, ax=axs[1][0])
    axs[1][0].set_title("T0 (K)")
    T2plot = axs[1][1].contourf(xs, ys, T2s, levels=100)
    fig.colorbar(T2plot, ax=axs[1][1])
    axs[1][1].set_title("T (K)")
    P02plot = axs[2][0].contourf(xs, ys, P02s, levels=100)
    fig.colorbar(P02plot, ax=axs[2][0])
    axs[2][0].set_title("P0 (kPa)")
    P2plot = axs[2][1].contourf(xs, ys, P2s, levels=100)
    fig.colorbar(P2plot, ax=axs[2][1])
    axs[2][1].set_title("P (kPa)")
    
    
    for ax in axs.ravel():
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/b")
    pass

def angle_grid(midspan:VelocityTriangle, root:VelocityTriangle, tip:VelocityTriangle, num, plot=False):
    """Plots the angle along the vane and blade, assuming they change linearly throughout the chord and quadratically throughout the span

    Args:
        midspan (VelocityTriangle): _description_
        root (VelocityTriangle): _description_
        tip (VelocityTriangle): _description_
        num (_type_): _description_
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Grid of x and y coordinates, relative to span and chord length
    xs = np.linspace(0,1,num,endpoint=True)
    xs,ys = np.meshgrid(xs,xs)
    
    # Reference coordinates
    x_ref = np.array([0,1]) # Leading and Trailing edge
    y_ref = np.array([0,0.5,1]) # Root, Midspan, Tip
    # Angles for vane (NOTE: Need to specify midspan for alpha1, since the velocity triangle process)
    z_ref = np.array([
            [midspan.alpha1, root.alpha2],
            [midspan.alpha1, midspan.alpha2],
            [midspan.alpha1, tip.alpha2]
    ]).T
    
    # Interpolation function
    vane_angles = intp.RectBivariateSpline(x_ref, y_ref,z_ref, kx=1, ky=2)
    
    # Angles for blade
    zr_ref = np.array([
            [root.alpha_r2, root.alpha_r3],
            [midspan.alpha_r2, midspan.alpha_r3],
            [tip.alpha_r2, tip.alpha_r3]   
    ]).T
    
    # Interpolation function
    blade_angles = intp.RectBivariateSpline(x_ref, y_ref, zr_ref, kx=1, ky=2)
    
    # Get the actual angles for grid of x,y
    vanes = vane_angles(xs, ys, grid=False)
    blades = blade_angles(xs, ys, grid=False)
    
    # Plot if needed
    if plot:
        fig, axs = plt.subplots(1,2)
        vane_plt = axs[0].contourf(xs, ys, np.rad2deg(vanes), levels=num)
        fig.colorbar(vane_plt, ax=axs[0])
        blade_plt = axs[1].contourf(xs, ys, np.rad2deg(blades), levels=num)
        fig.colorbar(blade_plt, ax=axs[1])

    return vanes, blades

def blade_surface_geometry(alphas: np.ndarray):
    """Get the blade surface geometry -- NOT WORKING

    Args:
        alphas (np.ndarray): _description_
    """
    ny, nx = alphas.shape
    
    xs = np.linspace(0,1,nx,endpoint=True)
    ys = np.linspace(0,1,ny,endpoint=True)
    
    # xs, ys = np.meshgrid(xs, ys)
    # zs = np.zeros_like(xs)
    # for i in range(len(zs)):
    #     print(xs[i])
    #     print(alphas[i])

def extract_final_conds(condition_array):
    """utility function"""
    ret_list = []
    for arg in condition_array:
        arg = arg.T[-1]
        ret_list.append(arg)
    return ret_list

# def relativizer(Us, alphas, alpha_rs, M2s, V2s, T0s, Ts, P0s, Ps, direction=1):
#     if direction == 1:
#         V2rs = (V2s*np.sin(alphas) - Us)/(np.sin(alpha_rs))
#     else:
#         V2rs = (V2s*np.sin(alphas) + Us)/(np.sin(alpha_rs))
#     M2rs = np.abs(M2s*(V2rs/V2s))
#     T0rs = temperature_ratio(M2rs)*Ts
#     P0rs = pressure_ratio(M2rs)*Ps
    
#     return M2rs, V2rs, T2s, T02s, P2s, P02s

def gridifier(*args, ax=1):
    """Takes arbitrary list of (same-length 1D) arrays of values and returns an N-dimensional mesh of all the values instead
    """
    ret_list = []
    for arg in args:
        if ax == 1:
            _, arg = np.meshgrid(arg, arg)
        else:
            arg, _ = np.meshgrid(arg, arg)
        ret_list.append(arg)
    return ret_list


def NANify(*args):
    """Fills all cells in all arrays with the NAN values form all other cells in all other arrays

    Returns:
        _type_: _description_
    """
    nan_array = np.full_like(args[0], 1)
    for arg in args:
        nan_array = np.where(np.isnan(arg), np.nan, nan_array)
    for arg in args:
        arg = np.where(np.isnan(nan_array), np.nan, arg)

    return args

def NANify_dict(d: dict):
    """Does the same as NANify but for a dictionary instead
    """
    items = list(d.items())
    nan_array = np.full_like(items[0], 1)
    for item in items:
        nan_array = np.where(np.isnan(item), np.nan, nan_array)
    for item in items:
        item = np.where(np.isnan(nan_array), np.nan, item)
    
    for i, key in enumerate(d):
        d[key] = items[i]


    return d

def new_method_midspan(Ma3s, alpha3s, AN2s, Uhubs, Rs):
    """Doing the entire process using the new method.
    All input parameters must either be SCALAR, or np.meshgrids() of the correct size

    Args:
        Ma3s (_type_): _description_
        alpha3s (_type_): _description_
        AN2s (_type_): _description_
        Uhubs (_type_): _description_
        Rs (_type_): _description_
    """

    # Static conditions at outlet, from computed total conditions in Part A, with meshgrid
    T3s = Conditions.T03/temperature_ratio(Ma3s)
    P3s = Conditions.P03/pressure_ratio(Ma3s)

    # Density, with meshgrid
    rho3s = P3s*1000/(T3s*R_air)

    # Velocity triangle at 3, absolute
    V3s = Ma3s*sound(T3s)
    Va3s = V3s*np.cos(alpha3s)
    Vt3s = V3s*np.sin(alpha3s)

    # Area from continuity, using total flow across blade (not including blade bleed air)
    A3s = Conditions.mdot_5i/(rho3s*Va3s)

    # Blade assumption ranges to get our RPMs
    # RPMs from area from continuity, get starting blade dimensions
    RPMs = np.sqrt(AN2s/A3s)
    rhs, rts = get_blade_from_RPM(RPMs, Uhubs, AN2s)
    rms = (rts + rhs) / 2

    # Blade velocities at meanline
    Ums = RPMs * (2*np.pi / 60) * rms

    # Tangengtial velocities at 3
    Vt3_rels = Vt3s + Ums

    # Relative velocities at 3
    alpha3_rels = np.atan2(Vt3_rels, Va3s)
    V3_rels = np.sqrt(Vt3_rels**2 + Va3s **2)

    ## Solving for Conditions at 2:
    # T1 is constant
    T1 = Conditions.T01/(temperature_ratio(Stage.M_in)) # Get your temperatures
    P1 = Conditions.P01/(pressure_ratio(Stage.M_in)) # Pressure at 1 also

    V1 = Stage.M_in * sound(T1)
    Va1 = V1*np.cos(Stage.swirl_in)
    rho1 = 1000 * P1 / (R_air*T1)
    A1 =  Conditions.mdot_5 / (rho1 *  Va1)
    
    # Temperature-based reaction
    T2s = T3s + (T1 - T3s)*Rs 
    M2s = np.sqrt(2*((Conditions.T02_mix/T2s)-1)/(0.333))
    V2s = M2s * sound(T2s)

    # Tangential velocity at 2
    Vt2s =  Conditions.w / Ums - Vt3s
    alpha2s = np.arcsin(Vt2s/V2s)
    Vt2_rels = Vt2s - Ums
    Va2s = np.cos(alpha2s)*V2s
    alpha2_rels = np.atan2(Vt2_rels, Va2s )

    Va2s = np.cos(alpha2s)*V2s  
    V2_rels = np.sqrt(Vt2_rels**2 + Va2s **2)

    rho2s = Conditions.mdot_5i / (V2s*np.cos(alpha2s) * A3s)
    

    P2s = rho2s * R_air * T2s / 1000
    P02s = P2s * pressure_ratio(M2s)

    # Estimated pressure loss coefficient obtained from the assumptions, across the vanes
    Yps = (Conditions.P01 - P02s)/(P02s - P2s)

    Ma3_rels = V3_rels / sound(T3s)
    P03_rels = P3s * pressure_ratio(Ma3_rels)
    T03_rels = T3s * temperature_ratio(Ma3_rels)

    Ma2_rels =  V2_rels / sound(T2s) 
    P02_rels = P2s * pressure_ratio(Ma2_rels)
    T02_rels = T2s * temperature_ratio(Ma2_rels)

    # Estimated pressure loss coefficient obtained from the assumptions, across the blades
    Yp_rels = (P02_rels - P03_rels) / (P03_rels - P3s)

    return  ({
        #Conditions at 1
        "P1": P1,
        "P01": Conditions.P01,
        "T1": T1,
        "T01": Conditions.T01,
        "rho1": rho1,
        "Va1": Va1,
        "V1": V1,   
        "A1": A1,
        # Conditions at 2
        "P2s":P2s,
        "P02s":P02s,
        "T2s":T2s,
        "T02s": Conditions.T02,
        "rho2s": rho2s,
        "Va2s": Va2s,   
        "Vt2s":Vt2s,
        "V2s":V2s,      
        "M2s":M2s,
        
        # Relative contitions at 2
        "V2_rels":V2_rels,
        "Vt2_rels": Vt2_rels,
        "Ma2_rels":Ma2_rels,
        "P02_rels":P02_rels,
        "T02_rels":T02_rels,
        
        
        # Conditions at 3
        "P3s":P3s,
        "P03s": Conditions.P03,
        "T3s":T3s,
        "T03s": Conditions.T03,
        "rho3s": rho3s,
        "Va3s":Va3s,
        "Vt3s": Vt3s,
        "V3s":V3s,
        "A3s":A3s,
        "Ma3s": Ma3s,
        
        # Relative conditions at 3
        "P03_rels":P03_rels,
        "T03_rels":T03_rels,
        "V3_rels":V3_rels,              
        "Vt3_rels":Vt3_rels,
        "Ma3_rels":Ma3_rels,

        
        # Blade angles and geometry
        "alpha1": Stage.swirl_in,
        "alpha2s":alpha2s,        
        "alpha2_rels":alpha2_rels,        
        "alpha3_rels":alpha3_rels,
        "alpha3s": alpha3s,
        "rhs":rhs,
        "rts":rts,
        "rms":rms,
        
        # Cycle characteristics
        "RPMs":RPMs,
        "Ums":Ums,        
        "Yps":Yps,        
        "Yp_rels":Yp_rels,
        "Rs": Rs      

    })


def yield_subgrids(arrays, axes, indices, values):
    """
    Extracts subgrids from the given meshgrid arrays by selecting the closest indices along specified axes.
    
    Parameters:
        arrays (list of np.ndarray): List of N-dimensional arrays to slice.
        axes (list of np.ndarray): Corresponding axis arrays that define the meshgrid.
        indices (list of int): Indices of axes to select.
        values (list of float): Target values to find the nearest match for in the specified axes.
    
    Returns:
        tuple: A tuple of subgrid arrays sliced at the nearest indices.
    """
    # Ensure inputs are valid
    assert len(indices) == len(values), "Indices and values lists must be of the same length."
    
    # Find the nearest index for each specified axis
    nearest_indices = []
    for axis, value in zip(indices, values):
        axis_values = axes[axis]
        diff = np.abs(axis_values - value)
        nearest_idx = np.unravel_index(np.argmin(diff), axis_values.shape)
        nearest_indices.append(nearest_idx[axis])  # Get index for the specific axis
    
    # Create slices for each dimension
    slices = [slice(None)] * arrays[0].ndim  # Initialize slices as full for all dimensions
    for axis, idx in zip(indices, nearest_indices):
        slices[axis] = idx  # Select only the nearest index for the specified axes
    
    # Apply slicing to all arrays
    subgrids = tuple(array[tuple(slices)] for array in arrays)
    
    return subgrids

def pretty_print(d, headers=0):
    """Prints big dictionaries to look nice (including embedded dictionaries)"""
    print(headers*"\t" + "{")
    for key in d:
        if type(d[key]) == dict:
            print((headers+1)*"\t" + f"{key}: ")
            pretty_print(d[key], headers=headers+1)
        elif "alpha" in key:
            print((headers+1)*"\t" + f"{key}:\t{np.rad2deg(d[key]):.4f} deg")
        else:
            print((headers+1)*"\t" + f"{key}:\t{d[key]:.4f}")
    print(headers*"\t" + "}")

def off_design(RPM, rm, V2, alpha2, beta2, alpha3_r, dP_rel, T2, P2, A3, V3r_actual):
    """Get the off-design values from RPM, conditions at 2, and relative pressure drop across blade
    
    KEY ASSUMPTIONS: Relative pressure drop from 2-3 remains the same (dP_rel = dP_rel (design point))
    Absolute flow properties at 2 are completely unaffected (T2=T2 (design), P2 = P2(design)) since the vane is unchanged by changing RPM

    Args:
        RPM (_type_): _description_
        rm (_type_): _description_
        V2 (_type_): _description_
        alpha2 (_type_): _description_
        beta2 (_type_): _description_
        alpha3_rel (_type_): _description_
        dP_rel (_type_): _description_
    """
    # Absolute and relative velocities at 2
    Um = RPM * (np.pi/30) * rm
    V2t = V2*np.sin(alpha2)
    V2t_r = V2t - Um
    V2a = V2*np.cos(alpha2)
    V2r = np.sqrt(V2t_r**2 + V2a**2)
    alpha2_r = np.atan2(V2t_r, V2a)
    incidence = alpha2_r - beta2
    
    # Relative flow properties at 2
    M2r = V2r / sound(T2)
    P02_rel = pressure_ratio(M2r)*P2
    T02_rel = temperature_ratio(M2r)*T2
    
    # DERIVATION:
    P03_rel = P02_rel - dP_rel

    
    #  rho3 * Va3 * A3 - mdot2 = 0
    continuity_eq = lambda v3r: (1000*P03_rel * (1 - (v3r**2) / (2*Cp_g*T2 + V2r**2)) ** (1.333/0.333)) / \
            ( R_air * (T2 + (V2r**2 - v3r**2) / (2*Cp_g)) ) * \
                (v3r * np.cos(alpha3_r)) * A3 - Conditions.mdot_5i


    V3r = fsolve(continuity_eq, V3r_actual,maxfev=1000)[0]
    T3 = T02_rel - V3r**2/(2*Cp_g)
    M3r = np.sqrt(2*(T02_rel/T3 - 1) / (0.333))
    P3 = P03_rel / (pressure_ratio(M3r))
    

    # print(1000*P03_rel * (1 - (V3r**2) / (2*Cp_g*T2 + V2r**2)) ** (1.333/0.333)) / \
    #         ( R_air * (T2 + (V2r**2 - V3r**2) / (2*Cp_g)))
            
    
    alpha3 = np.atan(np.tan(alpha3_r) - Um/(V3r*np.cos(alpha3_r)))
    V3 = V3r * np.cos(alpha3_r) / np.cos(alpha3)
    
    M3 = V3 / sound(T3)
    
    T03 = T3 * temperature_ratio(M3)
    P03 = P3 * temperature_ratio(M3)
    
    work = -1 * Cp_g * (T03 - Conditions.T02_mix) / 1000
    
    return {
        "Conditions at 2":{
            "alpha2": alpha2,
            "alpha2_r": alpha2_r,
            "V2r": V2r,
            "M2r": M2r,
            "P02_rel": P02_rel,
            "T02_rel": T02_rel   
        },
        "Conditions at 3": {
            "alpha3": alpha3,
            "alpha3_r": alpha3_r,
            "V3r": V3r,
            "M3r": M3r,
            "T3": T3,
            "P3": P3,
            "T03": T03,
            "P03": P03,
            "sp.work": work
        }
    }

def computation_from_losses(Yp, Yr, M3, alpha3, AN2, Uhub, RPM):
    """Computing the velocity triangles instead STARTING from losses, and assuming a fixed RPM.
    Requires solving a system of 5 non-linear equations and doesn't guarantee good results, so this method was not used.

    Args:
        Yp (_type_): _description_
        Yr (_type_): _description_
        M3 (_type_): _description_
        alpha3 (_type_): _description_
        AN2 (_type_): _description_
        Uhub (_type_): _description_
        RPM (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Params for later: (Yp, Yr, M3, alpha3, AN2, Uhub, RPM, area_ratio)
    
    Um = 0.5*Uhub + np.sqrt(np.pi*AN2/3600 +Uhub**2 / 4)
    A2 = AN2 / (RPM**2)

    T1 = Conditions.T01/temperature_ratio(Stage.M_in)
    P1 = Conditions.P01/pressure_ratio(Stage.M_in)
    rho1 = 1000 * P1 / (R_air*T1) 
    Va1 = Stage.M_in * sound(T1) * np.cos(Stage.swirl_in)
    A1 = Conditions.mdot_5 / (rho1 * Va1) 
    
    # Useful equations/constants to plug into the sovler
    mdot_1 = Conditions.mdot_5
    mdot_2 = lambda M2, alpha2: 1000*Conditions.P01 * np.sqrt(1.333/(R_air*Conditions.T02)) * ((temperature_ratio(M2))**(0.5) / pressure_ratio(M2)) / \
        ( 1 + Yp * (1 - 1/pressure_ratio(M2))) * M2 * np.cos(alpha2) * A2
    
    continuity_12 = lambda M2, alpha2: mdot_1 + \
        (Conditions.mdot_5i - Conditions.mdot_5) - \
        mdot_2(M2, alpha2)
        
    P03_from_relative = lambda M2, M2r, M3r: 1000*Conditions.P01 * \
        (pressure_ratio(M3)) / ((1 + Yr) * (1 + pressure_ratio(M3r))) * (pressure_ratio(M2r)/pressure_ratio(M3r)) * \
        ( 1 + Yp * (1 - 1/pressure_ratio(M2)))
        
    mdot_3 = lambda M2, M2r, M3r: P03_from_relative(M2, M2r, M3r) * \
        np.sqrt(1.333/(R_air*Conditions.T03)) * ((temperature_ratio(M3))**(0.5) / pressure_ratio(M3)) * \
        M3 * np.cos(alpha3) * A2
        
    continuity_23 = lambda M2, M2r, alpha2, M3r: mdot_2(M2, alpha2) - mdot_3(M2, M2r, M3r)
    
    # Finding an expression for M3r wrt Work Equation
    Vt2 = lambda M2, alpha2: M2*np.sin(alpha2) * sound(Conditions.T02_mix/temperature_ratio(M2))
    Vt3 = lambda M2, alpha2: Conditions.w / Um - Vt2(M2, alpha2)
    
    Va2 = lambda M2, alpha2: M2 * np.cos(alpha2) * sound(Conditions.T02_mix/temperature_ratio(M2))
    Va3 = M3 * np.cos(alpha3) * sound(Conditions.T03 / temperature_ratio(M3))
    
    V2r = lambda M2, alpha2: np.sqrt( (Vt2(M2, alpha2) - Um) **2 + Va2(M2, alpha2) ** 2)
    V3r = lambda M2, alpha2: np.sqrt( (Vt3(M2, alpha2) + Um) ** 2 + Va3 ** 2)
    
    M2r_from_work = lambda M2r, M2, alpha2 : M2r - V2r(M2, alpha2) / sound(Conditions.T02_mix/temperature_ratio(M2))
    M3r_from_work = lambda M3r, M2, alpha2 : M3r - V3r(M2, alpha2) / sound(Conditions.T03/temperature_ratio(M3))
    
    
    equation_vector = lambda M2, M2r, M3r, alpha2: [
        continuity_12(M2, alpha2),
        continuity_23(M2, M2r, alpha2, M3r),
        M2r_from_work(M2r, M2, alpha2),
        M3r_from_work(M3r, M2, alpha2)
    ]
    print(A2)
    equation_vector_packaged = lambda args: equation_vector(args[0], args[1], args[2], args[3])
    result_vector = least_squares(equation_vector_packaged, [1,1,1,1], bounds=((0,0,0,0), (1,1.5,1,np.pi)), ftol=None, xtol=None, max_nfev=1e4, jac='3-point')
    result_vector.x
    print(result_vector)
    print(equation_vector_packaged(result_vector.x))
    # result_vector = fsolve(equation_vector_packaged, [0,0.5,0.5,0], full_output=True)
    M2, Mr2, Mr3, alpha2 = result_vector.x
    # alpha2 = alpha2 % (2*np.pi)
    return M2, Mr2, Mr3, alpha2


def partDq2():
    """All the work for the HPT weight reduction/temperature increase trade study
    """
    # Initial values
    T03 = 515.992
    P04 = 592.845
    FHV = 40007.390 # kJ/kg
    T04 = 1132.865 # K
    mdot_5 = 6.797
    
    f_func = lambda dT: (Cp_g*(T04 + dT) - Cp_air*T03)/(FHV*1000 - Cp_g * (T04 + dT))
    
    dT_slope = 100 * 5/9 # 100F increaseC
    
    eta_h_func = lambda dT: 0.88 - (dT/dT_slope * 0.002)
    
    mdot_cool_func = lambda dT: 6.396 * 0.01 * (dT/dT_slope) # ADDITIONAL cooling air, assume supplied at blade outlet
    
    T05 = lambda dT: 910.521 + dT   
    T05_mix = lambda dT: 1/(Cp_g*(mdot_cool_func(dT) + mdot_5)) * (Cp_air * T03 * mdot_cool_func(dT) + Cp_g * mdot_5 * (905.929 + dT))
    
    P05 = lambda dT: (P04 * (1 - 1/eta_h_func(dT) * (1 - (T05(dT))/(1112.203 + dT)))**(1.333/0.333))
    
    P05_itd = lambda dT: (P05(dT))*0.99
        
    T06 = lambda dT: T05_mix(dT) * (1 - 0.91 * (1 - (105.939/P05_itd(dT))**(0.333/1.333)))
    
    W_net = lambda dT: -1 * (6.797 + mdot_cool_func(dT)) * (T06(dT) - T05_mix(dT)) * Cp_g
    
    
    
    SFC = lambda dT: 3600*f_func(dT) * mdot_5 / W_net(dT)


    dT = np.linspace(0, 300, 100, endpoint=True)
    SFCs = SFC(dT)
    mdots = mdot_cool_func(dT)
    etas = eta_h_func(dT)
    
    fs = f_func(dT)

    fig, axs = plt.subplots(2,2)
    
    axs[0][0].plot(dT, SFCs, "b-", linewidth=2 ,label="Change in SFC")
    axs[0][1].plot(dT, etas, "k-", linewidth=2,label="Change in HPT efficiency")
    axs[1][0].plot(dT, fs, "r-", linewidth=2 ,label="Change in fuel fraction")
    axs[1][1].plot(dT, mdots, "c-",linewidth=2 ,label="Change in cooling mass flow rate")

    for ax in axs.ravel():
        ax.legend()
        ax.set_xlim(0, max(dT))
        ax.set_xlabel("Temperature rise (K)")
    fig.suptitle("HPT Temperature Rise Trade Study")
    plt.show()
    
    
def pressure_distributions(M2r, M2m, M2t, M3r, M3m, M3t, P02):
    """Plotting the pressure distributions along the blade using hub, tip values where needed

    Args:
        M2r (_type_): _description_
        M2m (_type_): _description_
        M2t (_type_): _description_
        M3r (_type_): _description_
        M3m (_type_): _description_
        M3t (_type_): _description_
        P02 (_type_): _description_
    """
    NUM = 100
    span_fraction = np.linspace(0,1,NUM,endpoint=True)
    
    #P01 is constant varies according to RDTF
    P01s = np.full_like(span_fraction, Conditions.P01)
    P1s = P01s / pressure_ratio(Stage.M_in)
        
    # M2 varies, interpolate 
    M2_func = intp.interp1d([0, 0.5, 1], [M2r, M2m, M2t])
    M2s = M2_func(span_fraction)
    
    # Compute mixing, assuming flux is even across the entire region
    P2s = P02 / pressure_ratio(M2s)
    
    M3_func = intp.interp1d([0, 0.5, 1], [M3r, M3m, M3t])
    M3s = M3_func(span_fraction)
    
    P3s = Conditions.P03 / pressure_ratio(M3s)
        
    xs, ys = np.meshgrid([0,1], span_fraction)
    
    plt.set_cmap("jet")
    
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(16,9)
    vmin = np.min(np.array([P1s, P2s, P3s]).ravel())
    vmax = np.max(np.array([P1s, P2s, P3s]).ravel())


    ymax_vanes = span_fraction[np.argmax(P1s)], span_fraction[np.argmax(P2s)]
    ymax_blades = span_fraction[np.argmax(P2s)], span_fraction[np.argmax(P3s)]


    norm = Normalize(vmin, vmax)
    
    
    vaneplot = ax[0].contourf(xs, ys, np.array([P1s, P2s]).T, levels=20, vmin=vmin, vmax=vmax)
    ax[0].set_title("Vane pressure distribution")
    ax[0].set_xlabel("chord fraction")
    ax[0].set_ylabel("span fraction")
    
    bladeplot = ax[1].contourf(xs, ys, np.array([P2s, P3s]).T, levels=20, vmin=vmin, vmax=vmax)

    ax[1].set_title("Blade pressure distribution")
    ax[1].set_xlabel("chord fraction")
    ax[1].set_ylabel("span fraction")
    
    im = cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pressure (kPa)")
        
    plt.ylim(0,1)
    plt.show()
    
    pass  
    


def temperature_distributions(M2r, M2m, M2t, V2rr, V2rm, V2rt, V3rr, V3rm, V3rt):
    """Plotting the temperature distribution across the vane,blade using Vs where needed.

    Args:
        M2r (_type_): _description_
        M2m (_type_): _description_
        M2t (_type_): _description_
        V2rr (_type_): _description_
        V2rm (_type_): _description_
        V2rt (_type_): _description_
        V3rr (_type_): _description_
        V3rm (_type_): _description_
        V3rt (_type_): _description_
    """
    NUM = 100
    span_fraction = np.linspace(0,1,NUM,endpoint=True)
    
    #T01 varies according to RDTF
    T01s = Conditions.T01_var(span_fraction)
    T1s = T01s / temperature_ratio(Stage.M_in)
    V1s = Stage.M_in*sound(T1s) #Min is constant but V varies with temp
    
    # M2 varies, interpolate 
    M2_func = intp.interp1d([0, 0.5, 1], [M2r, M2m, M2t])
    M2s = M2_func(span_fraction)
    
    # Compute mixing, assuming flux is even across the entire region
    T2_pre = T01s / temperature_ratio(M2s)
    T2_post = (T2_pre * Cp_g * Conditions.mdot_5 + Conditions.T0_pre * Cp_air * (Conditions.mdot_5i-Conditions.mdot_5))/(Cp_g * Conditions.mdot_5i)
    T02_post = T2_post * temperature_ratio(M2s)
    
    # Use rothalpy conservation for next properties
    V2r_func = intp.interp1d([0, 0.5, 1], [V2rr, V2rm, V2rt])
    V2rs = V2r_func(span_fraction)
    T02rs = T2_post + V2rs**2 / (2*Cp_g)
    
    
    V3r_func = intp.interp1d([0, 0.5, 1], [V3rr, V3rm, V3rt])
    V3rs = V3r_func(span_fraction)
    
    T3s = T02rs - V3rs**2 / (2*Cp_g)
    
    xs, ys = np.meshgrid([0,1], span_fraction)
    
    plt.set_cmap("jet")
    
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(16,9)
    vmin = np.min(np.array([T1s, T2_pre, T2_post, T3s]).ravel())
    vmax = np.max(np.array([T1s, T2_pre, T2_post, T3s]).ravel())


    ymax_vanes = span_fraction[np.argmax(T1s)], span_fraction[np.argmax(T2_pre)]
    ymax_blades = span_fraction[np.argmax(T2_post)], span_fraction[np.argmax(T3s)]


    norm = Normalize(vmin, vmax)
    
    
    vaneplot = ax[0].contourf(xs, ys, np.array([T1s, T2_pre]).T, levels=20, vmin=vmin, vmax=vmax)
    ax[0].plot(np.array([0,1]), ymax_vanes, 'r--', label="spanwise location of maximum temperature")
    ax[0].legend()
    ax[0].set_title("Vane temperature distribution")
    ax[0].set_xlabel("chord fraction")
    ax[0].set_ylabel("span fraction")
    
    bladeplot = ax[1].contourf(xs, ys, np.array([T2_post, T3s]).T, levels=20, vmin=vmin, vmax=vmax)
    ax[1].plot(np.array([0,1]), ymax_blades, 'r--', label="spanwise location of maximum temperature")

    ax[1].set_title("Blade temperature distribution")
    ax[1].set_xlabel("chord fraction")
    ax[1].set_ylabel("span fraction")
    
    im = cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (K)")
        
    plt.ylim(0,1)
    plt.show()
    
    
    pass

def efficiency_from_yps(M2, M3r, V2, V3r, Yp, Yp_rel, alpha_r3, rt, rm, rh):
    """Get the actual efficiency from the pressure loss factors

    Args:
        M2 (_type_): _description_
        M3r (_type_): _description_
        V2 (_type_): _description_
        V3r (_type_): _description_
        Yp (_type_): _description_
        Yp_rel (_type_): _description_
        alpha_r3 (_type_): _description_
        rt (_type_): _description_
        rm (_type_): _description_
        rh (_type_): _description_

    Returns:
        _type_: _description_
    """
    zeta_N = Yp / (1 + 1.333/2 * M2**2)
    zeta_R = Yp_rel / (1 + 1.333/2 * M3r**2)
    print(zeta_N, zeta_R)


    eta_tt = 1 / (1 + ((zeta_N*V2**2 + zeta_R*V3r**2)/(2*Cp_g*(Conditions.T01 - Conditions.T03))))
    print(eta_tt)
    h = (rt - rh)
    delta_tc = 0.01 * h
    delta_eta_tt = 0.93 * eta_tt * (delta_tc) / (h * np.cos(alpha_r3)) * (rt/rm)
    print(delta_eta_tt)

    return eta_tt - delta_eta_tt

if __name__ == "__main__":
    """ -- Run things here, or comment them out to change how things are run. quit() early to kill the process before it finishes running
    """
    print(Conditions.T01_max)
    print(Conditions.T01_slope)
    # print(0.8*Blade.AN2_max/1e6)
    # print(0.95*Blade.rim_speed_max)
    # quit()
    # # Testing solver integrated with Yps    
    # Uhub = 0.95*Blade.rim_speed_max
    # AN2 = 0.95*Blade.AN2_max
    # RPM = 24000
    # Yp = 0.0582
    # Yr = 0.1598
    # M3 = 0.55
    # alpha3 = np.deg2rad(35) 
    # M2, Mr2, Mr3, alpha2 = computation_from_losses(Yp, Yr, M3, alpha3, AN2, Uhub, RPM)
    # print(M2, Mr2, Mr3, np.rad2deg(alpha2))
    var_dict = new_method_midspan(0.55, np.deg2rad(33), 0.95*Blade.AN2_max, 0.95*Blade.rim_speed_max, 0.60)
    var_dict1 = new_method_midspan(0.52, np.deg2rad(39), 0.80*Blade.AN2_max, 0.95*Blade.rim_speed_max, 0.525) # THESE ARE OUR VALUES
    V1 = var_dict1["V1"]

    pretty_print(var_dict1)

    # Yps = np.linspace(0.02, 0.15, num=100)
    # Yp_rels = np.linspace(0.15, 0.35, num=100)
    # Yps, Yp_rels = np.meshgrid(Yps, Yp_rels)
    
    #EFFICIENCIES
    eta_tts = efficiency_from_yps(var_dict1["M2s"], var_dict1["Ma3_rels"],
                              var_dict1["V2s"], var_dict1["V3_rels"],
                              0.0651, 0.1685, var_dict1['alpha3_rels'],
                              var_dict1["rts"],var_dict1["rms"], var_dict1["rhs"])
    print(eta_tts)
    
    print("OFF DESIGN")
    eta_tts = efficiency_from_yps(var_dict1["M2s"], 1.1520,
                              var_dict1["V2s"], 652.60,
                              0.0651, 0.1914, var_dict1['alpha3_rels'],
                              var_dict1["rts"],var_dict1["rms"], var_dict1["rhs"])
    print(eta_tts)    
    # rho3_computed = 1000 * var_dict1["P03_rels"] * (1 - (var_dict1["V3_rels"]**2) / (2*Cp_g*var_dict1['T2s'] + var_dict1["V2_rels"]**2)) ** (1.333/0.333) /(R_air * (var_dict1['T2s'] + (var_dict1["V2_rels"]**2 - var_dict1["V3_rels"]**2) / (2*Cp_g)))

    # ret_dict = off_design(0.9*var_dict1["RPMs"],
    #     rm=var_dict1["rms"],
    #     V2=var_dict1['V2s'],
    #     alpha2=var_dict1['alpha2s'],
    #     beta2=np.deg2rad(45.0854),
    #     alpha3_r=var_dict1["alpha3_rels"],
    #     dP_rel = var_dict1['P02_rels'] - var_dict1['P03_rels'],
    #     T2=var_dict1['T2s'],
    #     P2=var_dict1["P2s"],
    #     A3=var_dict1['A3s'],
    #     V3r_actual=var_dict1["V3_rels"])
    # pretty_print(ret_dict)
    
    # These are our triangles
    final_mean_triangle = VelocityTriangle(var_dict1["Ums"], V1, var_dict1["V2s"], var_dict1["V3s"], Stage.swirl_in, var_dict1["alpha2s"], np.deg2rad(39), var_dict1["alpha2_rels"], var_dict1["alpha3_rels"])
    # print(final_mean_triangle)
    final_root_triangle = get_offset_triangle(final_mean_triangle, var_dict1['rms'], var_dict1['rhs'])
    final_tip_triangle = get_offset_triangle(final_mean_triangle, var_dict1['rms'], var_dict1['rts'])
    print(f"Final Mean Triangle:{final_mean_triangle}\n\n")
    
    print(f"Final Root Triangle:{final_root_triangle}\n\n")
    print(f"Final Tip Triangle: {final_tip_triangle}\n\n")
    
    # Get all the other offset values
    R_root1 = get_temp_reaction(Conditions.T01, Conditions.T02, Conditions.T03,
                               final_root_triangle.V1, final_root_triangle.V2, final_root_triangle.V3)    
    R_tip1  = get_temp_reaction(Conditions.T01, Conditions.T02, Conditions.T03,
                               final_tip_triangle.V1, final_tip_triangle.V2, final_tip_triangle.V3)    

    M2_root1, M3_root1 = get_mach_offset(Conditions.T01, Conditions.T02, Conditions.T03,
                               final_root_triangle.V1, final_root_triangle.V2, final_root_triangle.V3)[1:]  
    
    M2_tip1, M3_tip1 = get_mach_offset(Conditions.T01, Conditions.T02, Conditions.T03,
                               final_tip_triangle.V1, final_tip_triangle.V2, final_tip_triangle.V3)[1:]
    
    root_V2r = final_root_triangle.V2 * np.cos(final_root_triangle.alpha2) / np.cos(final_root_triangle.alpha_r2)
    tip_V2r  = final_tip_triangle.V2 * np.cos(final_tip_triangle.alpha2) / np.cos(final_tip_triangle.alpha_r2)

    root_V3r = final_root_triangle.V3 * np.cos(final_root_triangle.alpha3) / np.cos(final_root_triangle.alpha_r3)
    tip_V3r  = final_tip_triangle.V3 * np.cos(final_tip_triangle.alpha3) / np.cos(final_tip_triangle.alpha_r3)    
    print("OFFSET RELATIVE")
    print(root_V2r)
    print(root_V3r)
    print(tip_V2r)
    print(tip_V3r)
    print("REACTIONS")  
    print(R_root1, R_tip1)
    print("MACHS")
    print(M2_root1, M2_tip1)
    print(M3_root1, M3_tip1)
    
    print("RELATIVE MACHS")
    print() #not useful
    
    # Temperature and Pressure distributions
    pressure_distributions(M2_root1, var_dict1['M2s'], M2_tip1,
                              M3_root1, var_dict1["Ma3s"], M3_tip1,var_dict1["P02s"])    
    temperature_distributions(M2_root1, var_dict1['M2s'], M2_tip1,
                              root_V2r, var_dict1["V2_rels"], tip_V2r,
                              root_V3r, var_dict1["V3_rels"], tip_V3r)


    # Some more analysis used for Dat's aero code
    Va_hub = final_root_triangle.V2*np.cos(final_root_triangle.alpha2)
    V2_rel = Va_hub/np.cos(final_root_triangle.alpha_r2)
    Mrel2_hub = V2_rel / sound(Conditions.T02_mix/temperature_ratio(M2_root1))
    print(f"!!{Mrel2_hub}")
    print(f"Root reaction = {R_root1}")
    print(f"Meanline = {final_mean_triangle}")
    print(f"Root = {final_root_triangle}")
    print(f"Tip = {final_tip_triangle}")
    print(root_V2r, tip_V2r)
    
    xs = np.linspace(0,2,5)
    ys = np.linspace(0,2,5)
    xs, ys = np.meshgrid(xs, ys)


    #### FULL 5-DIMENSIONAL PLOTTING PROCEDURE ###
    #### USING ALTERNATE PROCEDURE ####

    NUM = 100

    # Assume based on Ma3 and Alpha3s
    Ma3 = np.linspace(0.5, Stage.M_out_max*1.0, NUM, endpoint=True)
    alpha3 = np.linspace(np.deg2rad(35), Stage.swirl_out_max*1.0, NUM, endpoint=True)
    # Ma3 = np.array([0.55])
    # alpha3 = np.array([np.deg2rad(35)])

    # Manual setpoints
    # AN2 = Blade.AN2_max * 0.95
    # Uhub = Blade.rim_speed_max * 0.95
    AN2s = np.linspace(0.80*Blade.AN2_max, 1.0*Blade.AN2_max, 20, endpoint=True)
    Uhubs = np.linspace(0.80*Blade.rim_speed_max, 1.0*Blade.rim_speed_max, 20, endpoint=True)

    Rs = np.linspace(0.52, 0.53, 10, endpoint=True) # Assume a reaction
    Ma3s, alpha3s, AN2s, Uhubs, Rs = np.meshgrid(Ma3, alpha3, AN2s, Uhubs, Rs)
    mesh = [Ma3s, alpha3s, AN2s, Uhubs, Rs]

    var_dict = new_method_midspan(Ma3s, alpha3s, AN2s, Uhubs, Rs)
    V1 = var_dict["V1"]

    Yps = var_dict["Yps"]
    Yp_rels = var_dict["Yp_rels"]
    alpha2s = var_dict["alpha2s"]
    RPMs = var_dict["RPMs"]

    mean_tris = VelocityTriangle(var_dict["Ums"], V1, var_dict["V2s"], var_dict["V3s"], Stage.swirl_in, var_dict["alpha2s"], alpha3s, var_dict["alpha2_rels"], var_dict["alpha3_rels"])
    root_tris = get_offset_triangle(mean_tris, var_dict["rms"], var_dict["rhs"])
    tip_tris = get_offset_triangle(mean_tris, var_dict["rms"], var_dict["rts"])
    R_tips = get_temp_reaction(Conditions.T01, Conditions.T02, Conditions.T03,
                               tip_tris.V1, tip_tris.V2, tip_tris.V3)
    R_roots = get_temp_reaction(Conditions.T01, Conditions.T02, Conditions.T03,
                               root_tris.V1, root_tris.V2, root_tris.V3)   
    
    M2_roots = get_mach_offset(Conditions.T01, Conditions.T02, Conditions.T03,
                               root_tris.V1, root_tris.V2, root_tris.V3)[1]
    M2_tips = get_mach_offset(Conditions.T01, Conditions.T02, Conditions.T03,
                               tip_tris.V1, tip_tris.V2, tip_tris.V3)[1]

    # # M2_roots = np.where(M2_roots < 1, M2_roots, np.nan)
    # # Ums, V2s, V3s, alpha2s, alpha2_rels, alpha3_rels, P02s, P03_rels, Yps, Yp_rels, M2_roots, M2_tips, R_roots, R_tips = NANify(Ums, V2s, V3s, alpha2s, alpha2_rels, alpha3_rels, P02s, P03_rels, Yps, Yp_rels, M2_roots,  M2_tips, R_roots, R_tips)

    data = [Yps, Yp_rels, M2_roots, M2_tips, R_roots, R_tips, Ma3s, alpha3s, alpha2s, RPMs]
    mesh = [Ma3s, alpha3s, AN2s, Uhubs, Rs]

    AN2_val = 0.80*Blade.AN2_max
    U_val = 0.95*Blade.rim_speed_max
    R_val = 0.525

    Yps, Yp_rels, M2_roots, M2_tips, R_roots, R_tips, Ma3s,  alpha3s, alpha2s, RPMs = yield_subgrids(data, mesh, indices=(2,3,4), values=(AN2_val, U_val, R_val))

    # fig, axs = plt.subplots(1,2)
    # plt.set_cmap("jet")

    # fig.set_size_inches(16,9)
    # fig.suptitle(f"R={R_val}, AN^2={AN2_val/Blade.AN2_max*100:.2f}%, Uhub={U_val/Blade.rim_speed_max*100:.2f}%")
    # root_plot = axs[0].contourf(Ma3s, np.rad2deg(alpha3s), R_roots, levels=NUM)
    # axs[0].set_title("Root reaction")
    # plt.colorbar(root_plot, ax=axs[0])
    
    # tip_plot = axs[1].contourf(Ma3s, np.rad2deg(alpha3s), R_tips, levels=NUM)
    # axs[1].set_title("Tip reaction")
    # axs[0].grid()
    # axs[1].grid()
    # plt.colorbar(tip_plot, ax=axs[1])

    fig, axs = plt.subplots(1,2)
    plt.set_cmap("jet")

    fig.set_size_inches(9,4)
    fig.suptitle(f"R={R_val}, AN^2={AN2_val/Blade.AN2_max*100:.2f}%, Uhub={U_val/Blade.rim_speed_max*100:.2f}%")
    
    Yps = np.where(Yps > 0, Yps, np.nan)
    Yp_rels = np.where((Yp_rels > 0)&(Yp_rels < 5), Yp_rels, np.nan)
    Yps = np.where(np.isnan(Yp_rels), np.nan, Yps)
    Yp_rels = np.where(np.isnan(Yps), np.nan, Yp_rels)
    RPMs = np.where(np.isnan(Yps), np.nan, RPMs)
    M2_roots = np.where(np.isnan(Yps), np.nan, M2_roots)
    R_roots = np.where(np.isnan(M2_roots), np.nan, R_roots)
    
    plt1 = axs[0].contourf(Ma3s, np.rad2deg(alpha3s), Yps, levels=NUM)
    axs[0].set_title("Yp")
    cbar1 = plt.colorbar(plt1, ax=axs[0])
    cbar1.ax.plot([0,1], [var_dict1["Yps"], var_dict1["Yps"]], 'r')
    
    plt2 = axs[1].contourf(Ma3s, np.rad2deg(alpha3s), Yp_rels, levels=NUM)
    axs[1].set_title("Yp_rel")
    cbar2 = plt.colorbar(plt2, ax=axs[1])
    cbar2.ax.plot([0,1], [var_dict1["Yp_rels"], var_dict1["Yp_rels"]], 'r')

    fig2, axs2 = plt.subplots(1,2)
    fig2.set_size_inches(9,4)
    plt.set_cmap("jet")

    fig2.suptitle(f"R={R_val}, AN^2={AN2_val/Blade.AN2_max*100:.2f}%, Uhub={U_val/Blade.rim_speed_max*100:.2f}%")
    
    plt3 = axs2[0].contourf(Ma3s, np.rad2deg(alpha3s), R_roots, levels=NUM)
    axs2[0].set_title("Rroot")
    cbar1 = plt.colorbar(plt3, ax=axs2[0])
    cbar1.ax.plot([0,1], [R_root1, R_root1], 'r')
    
    
    plt4 = axs2[1].contourf(Ma3s, np.rad2deg(alpha3s), M2_roots, levels=NUM)
    axs2[1].set_title("M2_root")
    cbar2 = plt.colorbar(plt4, ax=axs2[1])
    cbar2.ax.plot([0,1], [M2_root1, M2_root1], 'r')
    
    
    # plt3 = axs[2].contourf(Ma3s, np.rad2deg(alpha3s), np.sign(Yp_rels), levels=1)
    # axs[2].set_title("Yp_rel")
    # plt.colorbar(plt3, ax=axs[2])
    
    axs[0].grid()
    axs[1].grid()
    axs2[0].grid()
    axs2[1].grid()
    mval, tval = 0.52, 39
    for ax in list(axs) + list(axs2):
        ax.set_xlabel("M3")
        ax.set_ylabel("alpha3 (deg)")
        ax.vlines(mval, 35, tval, 'r', linestyle="dashed")
        ax.hlines(tval, 0.50, mval, 'r', linestyle="dashed")
        ax.plot(mval, tval, 'ro')
        
    # axs[2].grid()
     

    # fig2, axs = plt.subplots(1,2)
    # fig2.set_size_inches(16,9)
    # fig2.suptitle(f"R={R}, AN^2={AN2/Blade.AN2_max*100:.2f}%, Uhub={Uhub/Blade.rim_speed_max*100:.2f}%")
    # print(np.rad2deg(alpha3s))
    # Yp_plot_vane = axs[0].contourf(Ma3s, np.rad2deg(alpha3s), Yps, levels=NUM)
    # axs[0].set_title("Yp vane")
    # axs[0].grid()
    # axs[1].grid() 
    # plt.colorbar(Yp_plot_vane, ax=axs[0])
    
    # Yp_plot_blade = axs[1].contourf(Ma3s, np.rad2deg(alpha3s), Yp_rels, levels=NUM)
    # axs[1].set_title("Yp blade")
    # plt.colorbar(Yp_plot_blade, ax=axs[1])


    # fig3, axs = plt.subplots(1,2)
    # fig3.set_size_inches(16,9)
    # fig3.suptitle(f"R={R}, AN^2={AN2/Blade.AN2_max*100:.2f}%, Uhub={Uhub/Blade.rim_speed_max*100:.2f}%")
    # print(np.rad2deg(alpha3s))
    # Mach_root_plot = axs[0].contourf(Ma3s, np.rad2deg(alpha3s), M2_roots, levels=NUM)
    # axs[0].set_title("Mach at root")
    # plt.colorbar(Mach_root_plot, ax=axs[0])
    
    # Mach_tip_plot = axs[1].contourf(Ma3s, np.rad2deg(alpha3s), M2_tips, levels=NUM)
    # axs[1].set_title("Mach at tip")
    # axs[0].grid()
    # axs[1].grid() 
    # plt.colorbar(Mach_tip_plot, ax=axs[1])



    # for ax in axs:
    #     ax.set_xlabel("Mach Number")
    #     ax.set_ylabel("alpha 3 (deg)")


    # fig4, axs = plt.subplots(1,3)
    # fig4.set_size_inches(16,9)
    # fig4.suptitle(f"R={R}, AN^2={AN2/Blade.AN2_max*100:.2f}%, Uhub={Uhub/Blade.rim_speed_max*100:.2f}%")
    # print(np.rad2deg(alpha3s))
    # alpha2_rel_plot = axs[0].contourf(Ma3s, np.rad2deg(alpha3s), np.rad2deg(alpha2_rels), levels=NUM)
    # axs[0].set_title("Blade inlet angle")
    # plt.colorbar(alpha2_rel_plot, ax=axs[0])
    
    # alpha3_rel_plot = axs[1].contourf(Ma3s, np.rad2deg(alpha3s), np.rad2deg(alpha3_rels), levels=NUM)
    # axs[1].set_title("Blade outlet angle")
    # plt.colorbar(alpha3_rel_plot, ax=axs[1])

    
    # alpha2_plot = axs[2].contourf(Ma3s, np.rad2deg(alpha3s), np.rad2deg(alpha2s), levels=NUM)
    # axs[2].set_title("Vane outlet angle")    
    # plt.colorbar(alpha2_plot, ax=axs[2])


    # axs[0].grid()
    # axs[1].grid() 
    # axs[2].grid()

    

    # for ax in axs:
    #     ax.set_xlabel("Mach Number")
    #     ax.set_ylabel("alpha 3 (deg)")

    # plot_triangle_reaction_ranges(triangle_midspan, 20000, rim_speeds, an2s)
    # plot_blades(rim_speed=Blade.rim_speed_max*0.95, AN2=Blade.AN2_max*0.95)
    
    # xs = np.linspace(0.15,3,100)
    # plt.plot(xs, func(xs))
    # plt.plot([0,3],[0,0], 'k--')

    


    # print(f"xs: {xs}")
    # print(f"ys: {ys}")

    plt.show()    