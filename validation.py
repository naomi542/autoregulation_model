from functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from main import *

CBF_lampe = pd.read_csv('CBF.csv', header=None)
cbf_Pa = CBF_lampe.iloc[:,0].to_numpy()
cbfs = CBF_lampe.iloc[:,1].to_numpy()

Va_lampe = pd.read_csv('AV.csv', header=None)
Va_Pa = Va_lampe.iloc[:,0].to_numpy()
Vas = Va_lampe.iloc[:,1].to_numpy()


def euler_next_state(x, x_dot, delta_t):
  """
  :param x: current state
  :param x_dot: state derivative
  :param delta_t: timestep
  :return: x_next: next state
  """
  x_next = x + x_dot * delta_t
  return x_next

def trapezoid_next_state(x, xa_dot, xb_dot, delta_t):
  """
  :param x: current state
  :param xa_dot: xa state derivative
  :param xb_dot: xb state derivative
  :param delta_t: timestep
  :return: x_next: next state
  """
  x_next = x + 0.5 * (xa_dot + xb_dot) * delta_t
  dx_dt = 0.5 * (xa_dot + xb_dot)
  return x_next, dx_dt

def Explicit_Trapezoid_Method_Ca(P_as, C_a, x):
    """
    :param tspan: time range list of beg and end times
    :param delta_t: stepsize for delta_t
    :param x0: initial states
    :return times: times (np.array)
    :return states: states (np.array)
    """

    C_a_vals = [C_a]
    dCa_dts = [0]
    for i, P_a in enumerate(P_as[:-1]):
        # Current
        C_a = C_a_vals[i]
        delta_P_a = P_as[i+1] - P_as[i]

        # Current derivative and next state using euler?
        xa_dot = arterial_compliance_change(C_a, x[i])
        eul_C_a_next = euler_next_state(C_a, xa_dot, delta_P_a)

        # Next derivative (euler) and next state using trap
        xb_dot = arterial_compliance_change(eul_C_a_next, x[i+1])
        C_a_next, dCa_dt = trapezoid_next_state(C_a, xa_dot, xb_dot, delta_P_a)
        C_a_vals.append(C_a_next)
        dCa_dts.append(dCa_dt)

    return np.asarray(C_a_vals), np.asarray(dCa_dts)

def Explicit_Trapezoid_Method_Pic(P_as, P_ic, C_as, dCa_dts, P_cs):
    """
    :param tspan: time range list of beg and end times
    :param delta_t: stepsize for delta_t
    :param x0: initial states
    :return times: times (np.array)
    :return states: states (np.array)
    """

    P_ic_vals = [P_ic]

    for i, P_a in enumerate(P_as[:-1]):
        # Current
        P_ic = P_ic_vals[i]
        delta_P_a = P_as[i+1] - P_as[i]

        # Current derivative and next state using euler?
        xa_dot = intercranial_pressure_change(P_a, P_cs[i], P_ic, C_as[i], dCa_dts[i])
        eul_P_ic_next = euler_next_state(P_ic, xa_dot, delta_P_a)

        # Next derivative (euler) and next state using trap
        xb_dot = intercranial_pressure_change(P_as[i+1], P_cs[i+1], eul_P_ic_next, C_as[i+1], dCa_dts[i+1])
        P_ic_next, _ = trapezoid_next_state(P_ic, xa_dot, xb_dot, delta_P_a)
        P_ic_vals.append(P_ic_next)

    return np.asarray(P_ic_vals)

def Explicit_Trapezoid_Method_simplified_Pic(P_as, P_ic, C_as, dCa_dts):
    """
    :param tspan: time range list of beg and end times
    :param delta_t: stepsize for delta_t
    :param x0: initial states
    :return times: times (np.array)
    :return states: states (np.array)
    """

    P_ic_vals = [P_ic]

    for i, P_a in enumerate(P_as[:-1]):
        # Current
        P_ic = P_ic_vals[i]
        delta_P_a = P_as[i+1] - P_as[i]

        # Current derivative and next state using euler?
        xa_dot = simplified_intercranial_pressure_change(P_a, P_ic, C_as[i], dCa_dts[i])
        eul_P_ic_next = euler_next_state(P_ic, xa_dot, delta_P_a)

        # Next derivative (euler) and next state using trap
        xb_dot = simplified_intercranial_pressure_change(P_as[i+1], eul_P_ic_next, C_as[i+1], dCa_dts[i+1])
        P_ic_next, _ = trapezoid_next_state(P_ic, xa_dot, xb_dot, delta_P_a)
        P_ic_vals.append(P_ic_next)

    return np.asarray(P_ic_vals)

def sigmoid_unit_test():
    norm_cbfs = np.arange(-1.0, 1.5, 0.01)
    sigmoids = []
    for x in norm_cbfs:
        sigmoids.append(static_sigmoidal_curve(x))

    plt.plot(norm_cbfs, sigmoids)
    plt.legend(["static sigmoidal function"])
    plt.xlabel("x (Normalized CBF deviation)")
    plt.show()
    return

def normalized_CBF_unit_test():
    xs = []
    for q in cbfs:
        xs.append(normalized_CBF_deviation(q))

    fig, axs = plt.subplots(2)
    axs[0].plot(cbf_Pa, cbfs)
    axs[0].legend("q (CBF)")
    axs[0].set_ylabel("mL")
    axs[1].plot(cbf_Pa, xs)
    axs[1].legend("x (normalized CBF)")
    plt.xlabel("P_a")
    plt.show()

def capillary_pressure_unit_test():
    C_a_initial = 0.15
    xs = []
    for q in cbfs:
        xs.append(normalized_CBF_deviation(q))

    C_a_vals, _ = Explicit_Trapezoid_Method_Ca(cbf_Pa, C_a_initial, xs) #using Pa from lampe with corresponding cbf

    P_ics = [] # using lampe
    R_as = []
    for P_a, C_a, V_a in zip(Va_Pa, C_a_vals, Vas): #calculate pics using Va from lampe and ca from above
        P_ic = P_a - (V_a/C_a)
        P_ics.append(P_ic)
        R_as.append((k_R * C_an**2 )/ V_a**2) #Ursino's Ra

    V_as_calculated = []
    for C_a, P_a, P_ic in zip(C_a_vals, Va_Pa, P_ics):
        V_a = arterial_arteriolar_volume(C_a, P_a, P_ic)
        V_as_calculated.append(V_a)

    # use our function to calculate pcs using lampes data
    P_cs = []
    for P_a, P_ic, R_a in zip(Va_Pa, P_ics, R_as):
        P_c = capillary_pressure(P_a, P_ic, R_a)
        P_cs.append(P_c)

    # get lampe's pc using lampe's q and Ra
    lampe_pc = []
    for P_a, q, R_a in zip(Va_Pa, cbfs, R_as):
        pc = (-1* q*R_a) + P_a
        lampe_pc.append(pc)

    fig, axs = plt.subplots(3)
    axs[0].plot(Va_Pa, P_ics)
    axs[0].plot(cbf_Pa, C_a_vals)
    axs[0].plot(Va_Pa, Vas)
    #axs[0].plot(Va_Pa, V_as_calculated)
    axs[0].legend(["P_ic", "C_a", "Lampe Va", "Our Va"])
    axs[0].set_ylabel("mmHg or mL")
    axs[1].plot(Va_Pa, P_cs)
    axs[1].plot(Va_Pa[3:], R_as[3:])
    axs[1].legend(["P_c", "R_a"])
    axs[1].set_ylabel("mmHg or mmHg^3*s/mL")
    axs[2].plot(Va_Pa, P_cs)
    axs[2].plot(Va_Pa[3:], lampe_pc[3:])
    axs[2].legend(["Our P_c", "lampe pc"])
    axs[2].set_ylabel("mmHg")
    plt.xlabel("P_a")
    plt.show()

def dCa_dt_unit_test():
    C_a_initial = 0.15
    P_ic_initial = 9.5
    xs = []
    for q in cbfs:
        xs.append(normalized_CBF_deviation(q))

    C_a_vals, dCa_dts = Explicit_Trapezoid_Method_Ca(cbf_Pa, C_a_initial, xs)

    fig, axs = plt.subplots(2)
    axs[0].plot(cbf_Pa, xs)
    axs[0].legend(["x (normalized CBF deviation)"])
    axs[0].set_ylabel("mL/s")
    axs[1].plot(cbf_Pa, C_a_vals)
    axs[1].plot(cbf_Pa, dCa_dts)
    axs[1].legend(["Compliance (C_a)"])
    axs[1].set_ylabel("mL/mmHg")
    plt.xlabel("P_a (mmHg)")
    plt.show()

def dPic_dt_unit_test():
    C_a_initial = 0.15
    P_ic_initial = 9.5
    xs = []
    for q in cbfs:
        xs.append(normalized_CBF_deviation(q))

    C_a_vals, dCa_dts = Explicit_Trapezoid_Method_Ca(cbf_Pa, C_a_initial, xs)

    P_ics = []
    R_as = []
    for P_a, C_a, V_a in zip(Va_Pa, C_a_vals, Vas):
        P_ic = P_a - (V_a/C_a)
        P_ics.append(P_ic)
        R_as.append((k_R * C_an**2 ) / V_a**2)

    # P_cs = []
    # for P_a, P_ic, R_a in zip(Va_Pa, P_ics, R_as):
    #     P_c = capillary_pressure(P_a, P_ic, R_a)
    #     P_cs.append(P_c)

    lampe_pc = []
    for P_a, q, R_a in zip(Va_Pa, cbfs, R_as):
        pc = (-1 * q * R_a) + P_a
        lampe_pc.append(pc)
    lampe_pc[0] = 0
    lampe_pc[1] = 0
    lampe_pc[2] = 0

    P_ics_calculated = Explicit_Trapezoid_Method_Pic(Va_Pa, P_ic_initial, C_a_vals, dCa_dts, lampe_pc)

    V_as_calculated = []
    for C_a, P_a, P_ic in zip(C_a_vals, Va_Pa, P_ics_calculated):
        V_a = arterial_arteriolar_volume(C_a, P_a, P_ic)
        V_as_calculated.append(V_a)

    plt.figure()
    plt.plot(Va_Pa, P_ics)
    plt.plot(Va_Pa, P_ics_calculated)
    plt.legend(["Lampe Pic", "Our P_ic"])
    plt.ylabel("mmHg")
    plt.xlabel("P_a (mmHg)")
    plt.show()

    plt.figure()
    plt.plot(Va_Pa, Vas)
    plt.plot(Va_Pa, V_as_calculated)
    plt.legend(["Lampe V_a", "V_a"])
    plt.xlabel("P_a")
    plt.show()
    return P_ics_calculated

def simplified_dPic_dt_unit_test():
    C_a_initial = 0.15
    P_ic_initial = 9.5
    xs = []
    for q in cbfs:
        xs.append(normalized_CBF_deviation(q))

    C_a_vals, dCa_dts = Explicit_Trapezoid_Method_Ca(cbf_Pa, C_a_initial, xs)

    P_ics = []
    for P_a, C_a, V_a in zip(Va_Pa, C_a_vals, Vas):
        P_ic = P_a - (V_a/C_a)
        P_ics.append(P_ic)

    P_ics_calculated = Explicit_Trapezoid_Method_simplified_Pic(Va_Pa, P_ic_initial, C_a_vals, dCa_dts)

    V_as_calculated = []
    for C_a, P_a, P_ic in zip(C_a_vals, Va_Pa, P_ics_calculated):
        V_a = arterial_arteriolar_volume(C_a, P_a, P_ic)
        V_as_calculated.append(V_a)

    plt.figure()
    plt.plot(Va_Pa, P_ics_calculated)
    plt.plot(Va_Pa, P_ics)
    plt.legend(["Our P_ic", "Lampe P_ic"])
    plt.ylabel("mmHg")
    plt.xlabel("P_a (mmHg)")
    plt.show()

    plt.figure()
    plt.plot(Va_Pa, Vas)
    plt.plot(Va_Pa, V_as_calculated)
    plt.legend(["Lampe V_a", "V_a"])
    plt.xlabel("P_a")
    plt.show()
    return P_ics_calculated

# Compare Ursino's resistance vs our resistance function using lampe data
def R_a_unit_test():
    R_as_ursino = []
    R_as_our_function =[]
    for V_a in Vas:  # calculate pics using Va from lampe and ca from above
        R_as_ursino.append((k_R * C_an ** 2) / V_a ** 2)  # Ursino's Ra
        lmda = alpha * np.sqrt(np.abs(V_a))
        our_ra= vascular_resistance(lmda, 0) # pass in 0 for Co2 cause that's what lampe data uses
        R_as_our_function.append(our_ra)
    plt.figure()
    plt.plot(Va_Pa[2:], R_as_ursino[2:])
    plt.plot(Va_Pa[2:], R_as_our_function[2:])
    plt.legend(["Ursino R_a", "Our_Ra"])
    plt.xlabel("P_a")
    plt.show()
    print(R_as_ursino)
    print(R_as_our_function)






# Function Calls

R_a_unit_test()
sigmoid_unit_test()
capillary_pressure_unit_test()
dCa_dt_unit_test()
full_Pics = dPic_dt_unit_test()
simplified_Pics = simplified_dPic_dt_unit_test()

plt.plot(Va_Pa, full_Pics)
plt.plot(Va_Pa, simplified_Pics)
plt.plot(Va_Pa, Vas)
plt.legend(["Full P_ic eqn", "Simplified P_ic", "Arterial Vol (V_a)"])
plt.xlabel("P_a (mmHg)")
plt.ylabel("Pressure (mmHg) and Volume (mL)")
plt.show()
