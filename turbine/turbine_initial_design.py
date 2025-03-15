import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Conversion constants
in_to_mm = 25.4
ft_to_m = in_to_mm*12/1000
Cp_g = 1148

# Container class for complete velocity triangle information for all stages
@dataclass
class VelocityTriangle:
    V1 : float
    V2 : float
    V3 : float
    alpha1 : float
    alpha2 : float
    alpha3 : float
    alpha_r2 : float
    alpha_r3 : float


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
    AN2_max = 45E10 * (in_to_mm/1000)**2 #(m**2)*(RPM**2)
    rim_speed_max = 1000 * ft_to_m #m/s
    
# Boundary conditions derived in Part A, Standard SI units unless noted otherwise 
class Conditions:
    T01 = 1132.865 # Turbine inlet temp
    T02 = 1132.865 # no work, adiabatic
    T02_mix = 1112.203 # After mixing with bleed air
    T03 = 910.521 # Turbine outlet temperature before mixing
    W_dot = (T02_mix-T03)*Cp_g
    mdot_5i = 6.729 # mdot going across the turbine blades, in kg/s
    w = W_dot/mdot_5i
    
    P01 = 592.845 # kPa
    P03 = 215.888 # kPa


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
def get_blade_from_RPM(n, rim_speed, AN2):
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

def plot_blades():
    # Plotting Blade RPM effect on blade radius
    blade_RPM_range = np.linspace(10000,25000,N) #RPM from 10k to 25k

    blade_rh_range, blade_rt_range = get_blade_from_RPM(blade_RPM_range, Blade.rim_speed_max, Blade.AN2_max)

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


def get_midspan_velocity_triangles(flow_coeff, reaction):

    ### Conditions at the inlet ###
    T1 = Conditions.T01/temperature_ratio(Stage.M_in) # Temperature at the first stage
    V1 = Stage.M_in*sound(T1) # Velocity at the first stage
    alpha1 = Stage.swirl_in


    Va = V1*np.cos(alpha1) # Axial velocity of first stage


    # Renaming all the variables to be more similar to our formula sheets....
    # Assuming Va = cst for initial design
    U = Va / flow_coeff
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

    alpha2 = np.atan(0.5*(W/(U*Va) - (R-1)/(0.5*phi)))
    alpha3 = np.atan(0.5*(W/(U*Va) + (R-1)/(0.5*phi)))

    ##### FINDING THE RELATIVE FLOW ANGLES #####
    # From the formulas
    # (1/phi) = tan(alpha_r3)-tan(alpha3) = tan(alpha2) - tan(alpha_r2) 

    alpha_r2 = np.atan(np.tan(alpha2) - 1/phi)    
    alpha_r3 = np.atan(np.tan(alpha3) + 1/phi)

    ##### FINDING THE STAGE VELOCITIES #####
    V2 = Va / np.cos(alpha2)
    V3 = Va / np.cos(alpha3)

    # Cast V1 to the same dimension as flow_coeff for easier plotting
    V1 = np.full_like(flow_coeff, V1)
    return_val = VelocityTriangle(V1,V2,V3,alpha1,alpha2,alpha3,alpha_r2,alpha_r3)

    return U, return_val

def analyze_triangle_range(flow_coeffs, reactions):
    flow_grid, reaction_grid = np.meshgrid(flow_coeffs, reactions)
    u, vel_tri = get_midspan_velocity_triangles(flow_grid, reaction_grid)

    fig, axs = plt.subplots(2,2)
    plt.set_cmap('jet')
    fig.set_size_inches(16,9)

    # Filter out values outside of the min and max ranges
    vel_tri.alpha3 = np.where((vel_tri.alpha3 >= Stage.swirl_out_min) & (vel_tri.alpha3 <= Stage.swirl_out_max), vel_tri.alpha3, np.nan)

    # For some reason all of the velocities are outside of the bounds? Doesn't really make sense but ok I guess???
    # vel_tri.V3 = np.where((vel_tri.V3 >= Stage.M_out_min*sound(Conditions.T03)) & (vel_tri.V3 <= Stage.M_out_max*sound(Conditions.T03)), vel_tri.V3, np.nan)
    print(f"Speed boundaries: {Stage.M_out_min*sound(Conditions.T03)}")
    # vel_tri.alpha3 = np.where(vel, vel_tri.alpha3, np.nan)

    vel_tri.V3 = np.where(np.isnan(vel_tri.alpha3), np.nan, vel_tri.V3)
    vel_tri.V2 = np.where(np.isnan(vel_tri.alpha3), np.nan, vel_tri.V2)
    u = np.where(np.isnan(vel_tri.alpha3), np.nan, u)

    N = 100
    u_contour = axs[0,0].contourf(flow_grid, reaction_grid, u, levels=N)
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


    

if __name__ == "__main__":

    flow_coeffs = np.arange(0.5, 0.8, 0.001)
    reactions = np.arange(0.35, 0.55, 0.001)

    # analyze_triangle_range(flow_coeffs, reactions)
    # print(get_midspan_velocity_triangles(0.6,0.45))
    plot_blades()



    plt.show()