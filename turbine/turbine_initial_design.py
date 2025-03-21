import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import interpolate as intp
from scipy.optimize import fsolve


# Conversion constants
in_to_mm = 25.4
ft_to_m = in_to_mm*12/1000
Cp_g = 1148 #J/kgK
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

def get_pressures(V1, P01, alphas, ARs):

    pass

def get_temperatures(V1, T01, alphas, ARs):
    pass

def get_mach(V1, M1, alphas, ARs):
    pass


def get_reaction(vel_tri: VelocityTriangle):


    phi = vel_tri.V2*np.cos(vel_tri.alpha2)/vel_tri.U
    return 0.5*phi*(np.tan(vel_tri.alpha_r3)-np.tan(vel_tri.alpha_r2))
    
def get_temp_reaction(T01s, T02s, T03s, V1s, V2s, V3s):
    T1s = T01s - (V1s)**2/(2*Cp_g)
    T2s = T02s - (V2s)**2/(2*Cp_g)
    T3s = T03s - (V3s)**2/(2*Cp_g)
    return (T2s - T3s)/ (T1s - T3s)


def get_mach_offset(T01s, T02s, T03s, V1s, V2s, V3s):
    T1s = T01s - (V1s)**2/(2*Cp_g)
    T2s = T02s - (V2s)**2/(2*Cp_g)
    T3s = T03s - (V3s)**2/(2*Cp_g)

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
    # Grid of x and y coordinates, relative to span and chord length
    xs = np.linspace(0,1,num,endpoint=True)
    xs,ys = np.meshgrid(xs,xs)
    
    # Reference coordinates
    x_ref = np.array([0,1]) # Leading and Trailing edge
    y_ref = np.array([0,0.5,1]) # Root, Midspan, Tip
    # Angles for vane (NOTE: Need to specify midspan for alpha1, since the velocity triangle process )
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
    ny, nx = alphas.shape
    
    xs = np.linspace(0,1,nx,endpoint=True)
    ys = np.linspace(0,1,ny,endpoint=True)
    
    # xs, ys = np.meshgrid(xs, ys)
    # zs = np.zeros_like(xs)
    # for i in range(len(zs)):
    #     print(xs[i])
    #     print(alphas[i])

def extract_final_conds(condition_array):
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
    ret_list = []
    for arg in args:
        if ax == 1:
            _, arg = np.meshgrid(arg, arg)
        else:
            arg, _ = np.meshgrid(arg, arg)
        ret_list.append(arg)
    return ret_list


def NANify(*args):
    nan_array = np.full_like(args[0], 1)
    for arg in args:
        nan_array = np.where(np.isnan(arg), np.nan, nan_array)
    for arg in args:
        arg = np.where(np.isnan(nan_array), np.nan, arg)

    return args

def NANify_dict(d: dict):
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
        NUM (int, optional): _description_. Defaults to 100.
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
    
    # Temperature-based reaction
    T2s = T3s + (T1 - T3s)*Rs 
    M2s = np.sqrt(2*((Conditions.T02/T2s)-1)/(0.333))
    V2s = M2s * sound(T2s)

    # Tangential velocity at 2
    Vt2s =  Conditions.w / Ums - Vt3s
    alpha2s = np.arcsin(Vt2s/V2s)
    Vt2_rels = Vt2s - Ums
    alpha2_rels = np.arcsin(Vt2_rels/V2s)  

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
        "T3s":T3s,
        "P3s":P3s,
        "V3s":V3s,
        "Va3s":Va3s,
        "Vt3s":Vt3s,
        "A3s":A3s,
        "RPMs":RPMs,
        "rhs":rhs,
        "rts":rts,
        "rms":rms,
        "Ums":Ums,
        "Vt3_rels":Vt3_rels,
        "alpha3_rels":alpha3_rels,
        "V3_rels":V3_rels,
        "T2s":T2s,
        "M2s":M2s,
        "V2s":V2s,
        "alpha2s":alpha2s,
        "alpha2_rels":alpha2_rels,
        "V2_rels":V2_rels,
        "P2s":P2s,
        "P02s":P02s,
        "Yps":Yps,
        "Ma3_rels":Ma3_rels,
        "P03_rels":P03_rels,
        "T03_rels":T03_rels,
        "Ma2_rels":Ma2_rels,
        "P02_rels":P02_rels,
        "T02_rels":T02_rels,
        "Yp_rels":Yp_rels,
    }, {
        "T1": T1,
    })

import numpy as np

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

if __name__ == "__main__":
        
    var_dict, const_dict = new_method_midspan(0.55, np.deg2rad(33), 0.95*Blade.AN2_max, 0.95*Blade.rim_speed_max, 0.6)
    V1 = Stage.M_in*sound(const_dict["T1"])
    
    print(var_dict)
    final_mean_triangle = VelocityTriangle(var_dict["Ums"], V1, var_dict["V2s"], var_dict["V3s"], Stage.swirl_in, var_dict["alpha2s"], np.deg2rad(35), var_dict["alpha2_rels"], var_dict["alpha3_rels"])
    # print(final_mean_triangle)

    xs = np.linspace(0,2,5)
    ys = np.linspace(0,2,5)
    xs, ys = np.meshgrid(xs, ys)
    
    # #### SECTION 1 ####
    # # Midspan velocity triangles
    
    # # Step 1: analysing reasonable ranges of flow coefficients and reactions to find our bounds
    # # flow_coeffs = np.arange(0.55,0.65, 0.001) # Phi
    # # reactions = np.arange(0.4,0.5,0.001) # Reaction
    # ## Velocity Ratio of 2.279 obtained form hand calcs as reasonable value
    # # analyze_triangle_range(flow_coeffs, reactions, 2.279)

    # # print(get_midspan_velocity_triangles(0.6,0.45))
    # T1_avg = Conditions.T01/temperature_ratio(Stage.M_in)
    # P1 = Conditions.P01/pressure_ratio(Stage.M_in)
    # V1_avg = Stage.M_in*sound(T1_avg)
    # A1 = (T1_avg*Conditions.mdot_5)/(V1_avg*P1)
    # RPM = np.sqrt(Blade.AN2_max*0.95/(A1/1.4))  
    
    
    # # These initial triangles are incorrect, they assume incompressible flow....
    # triangle_midspan = get_midspan_velocity_triangles(0.6, 0.48, 2.279)
    # rh, rt = get_blade_from_RPM(RPM, rim_speed=Blade.rim_speed_max*0.95, AN2 = Blade.AN2_max*0.95)
    # rm = (rh + rt)/2
    
    

    # rim_speeds = np.linspace(Blade.rim_speed_max*0.80, Blade.rim_speed_max, 100) # Rim speed needs to stay as high as possible
    # an2s = np.linspace(Blade.AN2_max*0, Blade.AN2_max, 100)
    # # plot_triangle_reaction_ranges(triangle_midspan, 20000, rim_speeds, an2s)

    
    # triangle_root = get_offset_triangle(triangle_midspan, rm, rh)
    # triangle_tip = get_offset_triangle(triangle_midspan, rm, rt)
    
    # print(triangle_root, triangle_midspan, triangle_tip, sep='\n'+50*'-'+'\n')
    # NUM = 11
    # vane_angles, blade_angles = angle_grid(triangle_midspan, triangle_root, triangle_tip, NUM)
    
    # Yp=0.1
    # AR=1.4
    

        
    
    # M1s = np.full_like(vane_angles,Stage.M_in) # Filled grid of M1 values
    # alpha1s = np.full_like(vane_angles, triangle_midspan.alpha1) # Filled grid of alpha1 values
    # Yps = np.linspace(0, Yp, num=NUM, endpoint=True) # Filled list of Yp values, assuming Yp linearly varies from 0 to max loss
    # Yps, _ = np.meshgrid(Yps, Yps) # Made into grid, loss increasing in chord
    
    # ARs = np.linspace(1,AR,num=NUM, endpoint=True) # Filled list of area ratio values, assuming AR varies linearly from 0 to max
    # ARs, _ = np.meshgrid(ARs, ARs) # Made into grid, AR increasing in chord
    
    # ys = np.linspace(0,1,NUM, endpoint=True)
    # T01s = Conditions.T01_var(ys)
    # _, T01s = np.meshgrid(T01s, T01s)
    # condition_array = full_solver(M1s, alpha1s, ARs, vane_angles, Yps, T01s)
    # condition_plotter(*condition_array, "Absolute Conditions along Vanes")

    # M2s, V2s, T2s, T02s, P2s, P02s = extract_final_conds(condition_array)

    # Rs = np.linspace(rh, rt, NUM, endpoint=True)
    # Us = RPM*(2*np.pi)/60 * Rs
    # # print(Us)

    # M2rs, V2rs, Trs, T0rs, Prs, P0s = relativizer(Us, vane_angles.T[-1], blade_angles.T[-1], M2s, V2s, T02s, T2s, P02s, P2s)
    
    
    # M2rs, T0rs  = gridifier(M2rs, T0rs)
    # # print(M2rs)
    # alpha2rs = blade_angles[0]
    # alpha2rs, _ = np.meshgrid(alpha2rs, alpha2rs)
    # ARs = np.full_like(blade_angles, 1)
    # condition_array = full_solver(M2rs, alpha2rs, ARs, blade_angles, Yps, T0rs)
    # # print(condition_array)
    # M3s, V3s, T3s, T03s, P3s, P03s = extract_final_conds(condition_array)

    

    # alpha3_t, alpha3_r = triangle_tip.alpha3, triangle_root.alpha3
    # alpha_r3t, alpha_r3r = triangle_tip.alpha_r3, triangle_root.alpha_r3
    # alpha3s = np.linspace(alpha3_r, alpha3_t, NUM, endpoint=True)
    # alpha_r3s = np.linspace(alpha_r3r, alpha_r3t, NUM, endpoint=True)

    # # Try un-relativizing everything
    # M3s, V3s, T3s, T03s, P3s, P03s = relativizer(Us, alpha_r3s, alpha3s, M3s, V3s, T03s, T3s, P03s, P3s, direction = -1)

    # condition_array = [M3s, V3s, T3s, T03s, P3s, P03s]

    # print(T03s)


    #### USING ANTHONY'S PROCEDURE ####

    NUM = 20

    # Assume based on Ma3 and Alpha3s
    Ma3 = np.linspace(Stage.M_out_min, Stage.M_out_max*1.0, NUM, endpoint=True)
    alpha3 = np.linspace(Stage.swirl_out_min, Stage.swirl_out_max*1.0, NUM, endpoint=True)
    # Ma3 = np.array([0.55])
    # alpha3 = np.array([np.deg2rad(35)])

    # Manual setpoints
    # AN2 = Blade.AN2_max * 0.95
    # Uhub = Blade.rim_speed_max * 0.95
    AN2s = np.linspace(0.80*Blade.AN2_max, 1.0*Blade.AN2_max, 20, endpoint=True)
    Uhubs = np.linspace(0.80*Blade.rim_speed_max, 1.0*Blade.rim_speed_max, 20, endpoint=True)

    Rs = np.linspace(0.4, 0.7, 10, endpoint=True) # Assume a reaction
    Ma3s, alpha3s, AN2s, Uhubs, Rs = np.meshgrid(Ma3, alpha3, AN2s, Uhubs, Rs)
    mesh = [Ma3s, alpha3s, AN2s, Uhubs, Rs]

    # # Blade velocities at meanline
    # Ums = RPMs * (2*np.pi / 60) * rms

    # Vt3_rels = Vt3s + Ums

    # alpha3_rels = np.atan2(Vt3_rels, Va3s)
    # V3_rels = np.sqrt(Vt3_rels**2 + Va3s **2)


    # ## Solving for Conditions at 2:
    # T1 = Conditions.T01/(temperature_ratio(Stage.M_in)) # Get your temperatures
    
    # T2s = T3s + (T1 - T3s)*R # 
    # M2s = np.sqrt(2*((Conditions.T02/T2s)-1)/(0.333))
    # V2s = M2s * sound(T2s)

    # Vt2s =  Conditions.w / Ums - Vt3s
    # alpha2s = np.arcsin(Vt2s/V2s)
    # Vt2_rels = Vt2s - Ums
    # alpha2_rels = np.arcsin(Vt2_rels/V2s)  

    # Va2s = np.cos(alpha2s)*V2s  
    # V2_rels = np.sqrt(Vt2_rels**2 + Va2s **2)

    # rho2s = Conditions.mdot_5i / (V2s*np.cos(alpha2s) * A3s)

    # P2s = rho2s * R_air * T2s / 1000
    # P02s = P2s * pressure_ratio(M2s)


    # Yps = (Conditions.P01 - P02s)/(P02s - P2s)

    # Ma3_rels = sound(T3s) / V3_rels
    # P03_rels = P3s * pressure_ratio(Ma3_rels)
    # T03_rels = T3s * temperature_ratio(Ma3_rels)

    # Ma2_rels = sound(T2s) / V2_rels
    # P02_rels = P3s * pressure_ratio(Ma2_rels)
    # T02_rels = T3s * temperature_ratio(Ma2_rels)

    # Yp_rels = (P02_rels - P03_rels) / (P03_rels - P3s)
    var_dict, const_dict = new_method_midspan(Ma3s, alpha3s, AN2s, Uhubs, Rs)
    V1 = Stage.M_in*sound(const_dict["T1"])

    Yps = var_dict["Yps"]
    Yp_rels = var_dict["Yp_rels"]

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

    data = [Yps, Yp_rels, M2_roots, M2_tips, R_roots, R_tips, Ma3s, alpha3s]
    mesh = [Ma3s, alpha3s, AN2s, Uhubs, Rs]

    AN2_val = 0.95*Blade.AN2_max
    U_val = 0.95*Blade.rim_speed_max
    R_val = 0.6

    Yps, Yp_rels, M2_roots, M2_tips, R_roots, R_tips, Ma3s,  alpha3s = yield_subgrids(data, mesh, indices=(2,3,4), values=(AN2_val, U_val, R_val))

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

    fig.set_size_inches(16,9)
    fig.suptitle(f"R={R_val}, AN^2={AN2_val/Blade.AN2_max*100:.2f}%, Uhub={U_val/Blade.rim_speed_max*100:.2f}%")
    root_plot = axs[0].contourf(Ma3s, np.rad2deg(alpha3s), R_roots, levels=NUM)
    axs[0].set_title("Root reaction")
    plt.colorbar(root_plot, ax=axs[0])
    
    tip_plot = axs[1].contourf(Ma3s, np.rad2deg(alpha3s), R_tips, levels=NUM)
    axs[1].set_title("Tip reaction")
    axs[0].grid()
    axs[1].grid()    
    plt.colorbar(tip_plot, ax=axs[1])

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