from functions import *
from constants import *
import numpy as np

def sim1_dynamics(x, P_CO2, sim_num):
    """
    Dynamics for sim 1
    Recreating Lampe's simulation
    :param x: state & input vector [C_a, P_ic, P_a]
    :param P_CO2: CO2 partial pressure
    :return: state derivative vector
    """
    C_a = x[0]
    P_ic = x[1]
    P_a = x[2]

    # Get all the variables that depend on state vars
    V_a = arterial_arteriolar_volume(C_a, P_a, P_ic)        # arterial-arteriolar volume
    lmda = alpha * np.sqrt(V_a)                             # coefficient of vascular radii
    R_a = vascular_resistance(lmda, P_CO2)                  # arterial resistance
    P_c = capillary_pressure(P_a, P_ic, R_a)                # capillary pressure
    q = get_CBF(P_a, P_c, R_a)                              # CBF
    x = normalized_CBF_deviation(q)                         # normalized CBF

    # State derivative equations
    dCa_dt = arterial_compliance_change(C_a, x)
    if sim_num==1:
        dP_ic_dt = intercranial_pressure_change(P_a, P_c, P_ic, C_a, dCa_dt)
    elif sim_num ==3:
        dP_ic_dt = simplified_intercranial_pressure_change(P_a, P_ic, C_a, dCa_dt)

    return [dCa_dt, dP_ic_dt, dPa_dt]

def sim2_dynamics(x, P_vs):
    """
    Dynamics for sim 1
    Recreating Lampe's simulation
    :param x: state & input vector [C_a, P_ic, P_a]
    :param P_vs: dural sinus pressure
    :return: state derivative vector
    """

    C_a = x[0]
    P_ic = x[1]
    P_a = x[2]

    # Get all the variables that depend on state vars
    V_a = arterial_arteriolar_volume(C_a, P_a, P_ic)        # arterial-arteriolar volume
    lmda = alpha * np.sqrt(V_a)                             # coefficient of vascular radii
    R_a = vascular_resistance(lmda)                         # arterial resistance
    P_c = capillary_pressure(P_a, P_ic, R_a)                # capillary pressure
    q = get_CBF(P_a, P_c, R_a)                              # CBF
    x = normalized_CBF_deviation(q)                         # normalized CBF

    # State derivative equations
    dCa_dt = arterial_compliance_change(C_a, x)
    dP_ic_dt = intercranial_pressure_change(P_a, P_c, P_ic, C_a, dCa_dt, P_vs)

    return [dCa_dt, dP_ic_dt, dPa_dt]
