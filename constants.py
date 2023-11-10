import pandas as pd

# Level, Number of Vessels, Length [mm], D0 (vascular diameter at baseline) [mm], Reactivity [%/mmHg], 1-Y [%]
vascular_reactivity_data = pd.read_csv("vascular_reactivity_data.csv", names=['Level', 'Number of Vessels', 'Length', 'Diameter', 'Reactivity', '1-Y'])

# Other constants
P_vs_const = 6.0                    # const dural sinus pressure [mmHg] used in sim1 and sim3
P_CO2_const = 36.0                  # const CO2 partial pressure [mmHg] used in sim2
G = 1.5                             # maximum autoregulation gain [mL/mmHg]
delta_C_a1 = 0.75                   # amplitude of sigmoidal curve if x <= 0
delta_C_a2 = 0.075                  # amplitude of sigmoidal curve if x > 0
C_an = 0.15                         # central value of the sigmoidal curve [mL/mmHg]
R_f = 2380.0                        # resistance to formation of CSF [mmHg*s/mL]
R_pv = 1.24                         # proximal venous resistance [mmHg*s/mL]
R_o = 526.3                         # resistance to outflow of CSF [mmHg*s/mL]
I_i = 1/30.0                        # injection rate of CSF [mL/s]
q_n = 12.5                          # value of CBF required by tissue metabolism [mL/s]
T = 20.0                            # time constant of regulation [s]
alpha = 0.2                         # fitting constant
k_E = 0.11                          # [1/mL]
k_R = 49100.0                       # [mmHg^3*s/mL]
mu = 0.000026252155                 # [mmHg*s] average blood viscosity at 37C

# Parameters
P_CO2_sims = [0, 10, 20, 30, 40]    # partial pressure of CO2 for sim1 + sim3 [mmHg]
P_vs_sims = [5, 10, 15, 25, 35]     # dural sinus pressures values for sim2 [mmHg]

dPa_dt = 2/3.0                      # change in P_a over time [mmHg/s]