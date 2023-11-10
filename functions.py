from constants import *
import numpy as np

# Equation 18 in Lampe paper
def normalized_CBF_deviation(q):
  """
  Return normalized CBF deviation (x) given current CBF (q)
  :param q:     current CBF
  :return:      normalized CBF deviation (x)
  """
  return (q-q_n)/q_n

# Equation 20 in Lampe paper
def arterial_arteriolar_volume(C_a, P_a, P_ic):
  """
  :param C_a:   arterial compliance
  :param P_a:   arterial pressure
  :param P_ic:  ICP
  :return:      arterial-arteriolar volume (V_a)
  """
  #V_a = C_a*(P_a - P_ic) if (P_a - P_ic) > 0 else 0
  return np.abs(C_a*(P_a - P_ic)) * 0.85

# Equation 21 in Lampe paper
def capillary_pressure(P_a, P_ic, R_a):
  """
  :param C_a:   arterial compliance
  :param P_a:   arterial pressure
  :param P_ic:  ICP
  :param R_a:   arterial resistance
  :return:      capillary pressure (P_c)
  """
  return (P_a*R_pv + P_ic*R_a)/(R_pv + R_a)

# Equation 4 - CBF (eq 23a in Lampe paper)
def get_CBF(P_a, P_c, R_a):
  """
  :param P_a:     arterial pressure
  :param P_c:     capillary pressure
  :param R_a:     arterial resistance
  :return:        CBF (q)
  """
  #q = (P_a - P_c)/R_a if (P_a - P_c) > 0 else 0
  return (P_a - P_c)/R_a

# Equation 1 - Arterial Compliance ODE (eq 16 in Lampe paper)
def arterial_compliance_change(C_a, x):
  """
  :param P_a:     arterial pressure
  :param P_c:     capillary pressure
  :param V_a:     arterial-arteriolar volume
  :param P_CO2:   carbon dioxide partial pressure
  :param C_a:     arterial compliance
  :return:        change in arterial compliance (dC_a/dt)
  """
  return  (1 / T) * (static_sigmoidal_curve(x) - C_a)

# Equation 2 - Sigmoidal Static Function (eq 24 in Lampe paper)
def static_sigmoidal_curve(x):
  """
  :param x:       normalized CBF deviation
  :return:        tbh I'm not sure - some sigmoid function describing x lol
  """
  delta_C = delta_C_a1 if x <= 0 else delta_C_a2      # amplitude of sigmoidal curve
  k_sigma = delta_C/4.0                               # central slope
  exponential = np.exp(G*x/k_sigma)                  # exponential term
  return ((C_an + delta_C/2) + (C_an - delta_C/2) * exponential) / (1 + exponential)


# Equation 3
def intercranial_pressure_change(P_a, P_c, P_ic, C_a, dCa_dt, P_vs=P_vs_const):
  """
    :param P_a    :     arterial pressure
    :param P_c    :     capillary pressure
    :param P_ic   :     ICP
    :param C_a    :     arterial compliance
    :param V_a    :     arterial-arteriolar volume
    :param P_CO2  :     pass in specific partial pressure of CO2 [mmHg] - constant
    :return       :     change in intercranial pressure (dP_ic/dt)
  """
  C_ic = (1 + C_a * k_E * P_ic) / (k_E * P_ic)
  # C_ic = (1 + C_a) / (k_E * P_ic)
  CSF_mechanism = (P_c - P_ic)/R_f - (P_ic - P_vs)/R_o + I_i
  dVa_dt = C_a*(dPa_dt - P_ic) + dCa_dt*(P_a - P_ic)
  return (dVa_dt + CSF_mechanism) / C_ic

# Equation 7
def simplified_intercranial_pressure_change(P_a, P_ic, C_a, dCa_dt):
  """
  :param P_a      arterial pressure
  :param P_ic     intercranial pressure
  :param C_a      arterial compliance
  """
  C_ic = (1 + C_a * k_E * P_ic) / (k_E * P_ic)
  # C_ic = (1 + C_a) / (k_E * P_ic)
  dVa_dt = C_a*(dPa_dt - P_ic) + dCa_dt * (P_a - P_ic)
  return dVa_dt / C_ic

# Equation 5 - Piechnik equation for vascular resistance
def vascular_resistance(lmda, P_CO2=P_CO2_const):
  """
  :param lmda:        fitting constant - alpha * sqrt(V_a)
  :param P_CO2:       carbon dioxide partial pressure
  :return:            vascular resistance
  """

  # summing up all of the values from each level to determine RA
  RA = 0
  for x in range (len(vascular_reactivity_data)):
    # get respective variable from vascular reactivity data for each level
    m = vascular_reactivity_data['Number of Vessels'].iloc[x]
    l = vascular_reactivity_data['Length'].iloc[x] #mm
    r = vascular_reactivity_data['Diameter'].iloc[x]/2.0 #mm
    c = vascular_reactivity_data['Reactivity'].iloc[x]

    l = l/20
    r = r/100
    term = (8 * mu * l) / ((np.pi) * m)  * (lmda * r * (1 + c * P_CO2))**-4
    # term = (8 * mu) / (np.pi * m * l) * (lmda * r**-1 * (1 + c * P_CO2))**-4
    RA += term
  return RA 

