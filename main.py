from functions import *
from dynamics import *
from constants import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

def simulation1(C_a_initial, P_ic_initial, T, P_CO2, sim_num):
  #P_CO2 = P_CO2_sims[0]
  P_a_initial = 0

  def f(t, x):
    return sim1_dynamics(x, P_CO2, sim_num)

  sol = solve_ivp(f, [0, T], [C_a_initial, P_ic_initial, P_a_initial], max_step=1)
  time = sol.t
  C_a_vals = sol.y[0, :]
  P_ic_vals = sol.y[1, :]
  P_a_vals = sol.y[2, :]

  CBF = []
  for C_a, P_ic, P_a in zip(C_a_vals, P_ic_vals, P_a_vals):
    V_a = arterial_arteriolar_volume(C_a, P_ic, P_a)        # arterial-arteriolar volume
    lmda = alpha * np.sqrt(V_a)                             # coefficient of vascular radii
    R_a = vascular_resistance(lmda, P_CO2)                  # arterial resistance
    P_c = capillary_pressure(P_a, P_ic, R_a)                # capillary pressure
    q = get_CBF(P_a, P_c, R_a)                              # CBF
    x = normalized_CBF_deviation(q)                         # normalized CBF
    CBF.append(q)
  return P_a_vals, CBF

def simulation2(C_a_initial, P_ic_initial, P_vs, T=300):
  P_a_initial = 0

  def f(t, x):
    return sim2_dynamics(x, P_vs)

  sol = solve_ivp(f, [0, T], [C_a_initial, P_ic_initial, P_a_initial], rtol=1e-5, atol=1e-8)
  time = sol.t
  C_a_vals = sol.y[0, :]
  P_ic_vals = sol.y[1, :]
  P_a_vals = sol.y[2, :]

  CBF = []
  for C_a, P_ic, P_a in zip(C_a_vals, P_ic_vals, P_a_vals):

    V_a = arterial_arteriolar_volume(C_a, P_ic, P_a)        # arterial-arteriolar volume
    lmda = alpha * np.sqrt(np.abs(V_a))                            # coefficient of vascular radii
    R_a = vascular_resistance(lmda)                  # arterial resistance
    #R_a = (k_R * C_an ** 2) / V_a ** 2
    P_c = capillary_pressure(P_a, P_ic, R_a)                # capillary pressure
    q = get_CBF(P_a, P_c, R_a)                              # CBF
    x = normalized_CBF_deviation(q)                         # normalized CBF
    CBF.append(q)
  return C_a_vals, P_ic_vals, P_a_vals, CBF

# Figure 9 lampe
def fig_9_lampe():
  #### lampe results ####
  CBF_lampe_0 = pd.read_csv('0.csv', header=None)
  cbf_0_Pa = CBF_lampe_0.iloc[:, 0].to_numpy()
  cbfs_0 = CBF_lampe_0.iloc[:, 1].to_numpy()

  CBF_lampe_10 = pd.read_csv('10.csv', header=None)
  cbf_10_Pa = CBF_lampe_10.iloc[:, 0].to_numpy()
  cbfs_10 = CBF_lampe_10.iloc[:, 1].to_numpy()

  CBF_lampe_20 = pd.read_csv('20.csv', header=None)
  cbf_20_Pa = CBF_lampe_20.iloc[:, 0].to_numpy()
  cbfs_20 = CBF_lampe_20.iloc[:, 1].to_numpy()

  CBF_lampe_30 = pd.read_csv('30.csv', header=None)
  cbf_30_Pa = CBF_lampe_30.iloc[:, 0].to_numpy()
  cbfs_30 = CBF_lampe_30.iloc[:, 1].to_numpy()

  CBF_lampe_40 = pd.read_csv('40.csv', header=None)
  cbf_40_Pa = CBF_lampe_40.iloc[:, 0].to_numpy()
  cbfs_40 = CBF_lampe_40.iloc[:, 1].to_numpy()

  #### our results #####
  cbf_ours_0_Pa = cbf_0_Pa
  cbfs_ours_0 = np.zeros(cbf_0_Pa.size)

  CBF_ours_10 = pd.read_csv('10_ours.csv', header=None)
  cbf_ours_10_Pa = CBF_ours_10.iloc[:, 0].to_numpy()
  cbfs_ours_10 = CBF_ours_10.iloc[:, 1].to_numpy()

  CBF_ours_20 = pd.read_csv('20_ours.csv', header=None)
  cbf_ours_20_Pa = CBF_ours_20.iloc[:, 0].to_numpy()
  cbfs_ours_20 = CBF_ours_20.iloc[:, 1].to_numpy()

  CBF_ours_30 = pd.read_csv('30_ours.csv', header=None)
  cbf_ours_30_Pa = CBF_ours_30.iloc[:, 0].to_numpy()
  cbfs_ours_30 = CBF_ours_30.iloc[:, 1].to_numpy()

  CBF_ours_40 = pd.read_csv('40_ours.csv', header=None)
  cbf_ours_40_Pa = CBF_ours_40.iloc[:, 0].to_numpy()
  cbfs_ours_40 = CBF_ours_40.iloc[:, 1].to_numpy()

  ##### RMSE #######
  rmse= []
  norm = max(np.max(cbfs_0), np.max(cbfs_ours_0))
  rmse.append(mean_squared_error(cbfs_0, cbfs_ours_0,squared=False)) #y_true (lampes), #y_pred (ours)
  norm = max(np.max(cbfs_10), np.max(cbfs_ours_10))
  rmse.append(mean_squared_error(cbfs_10, cbfs_ours_10,squared=False)) #10
  norm = max(np.max(cbfs_20), np.max(cbfs_ours_20))
  rmse.append(mean_squared_error(cbfs_20, cbfs_ours_20,squared=False)) #20
  norm = max(np.max(cbfs_30), np.max(cbfs_ours_30))
  rmse.append(mean_squared_error(cbfs_30, cbfs_ours_30,squared=False)) #30
  norm = max(np.max(cbfs_40), np.max(cbfs_ours_40))
  rmse.append(mean_squared_error(cbfs_40, cbfs_ours_40,squared=False)) #40

  #####R_SQUARED###########
  r_squared=[]
  r_squared.append(r2_score(cbfs_0, cbfs_ours_0))  # y_true, y_pred
  r_squared.append(r2_score(cbfs_10, cbfs_ours_10)) #10
  r_squared.append(r2_score(cbfs_20, cbfs_ours_20)) #20
  r_squared.append(r2_score(cbfs_30, cbfs_ours_30)) #30
  r_squared.append(r2_score(cbfs_40, cbfs_ours_40)) #40


  ##########PLOT US AND LAMPE TOGETHER##############
  plt.figure()
  plt.plot(cbf_0_Pa, cbfs_0, 'r',linestyle='dashed')
  plt.plot(cbf_10_Pa, cbfs_10, '#5CB200',linestyle='dashed')
  plt.plot(cbf_20_Pa, cbfs_20, 'b',linestyle='dashed')
  plt.plot(cbf_30_Pa, cbfs_30, 'orchid',linestyle='dashed')
  plt.plot(cbf_40_Pa, cbfs_40, 'skyblue',linestyle='dashed')

  plt.plot(cbf_ours_0_Pa, cbfs_ours_0, 'r')
  plt.plot(cbf_ours_10_Pa, cbfs_ours_10, '#5CB200')
  plt.plot(cbf_ours_20_Pa, cbfs_ours_20, 'b')
  plt.plot(cbf_ours_30_Pa, cbfs_ours_30, 'orchid')
  plt.plot(cbf_ours_40_Pa, cbfs_ours_40, 'skyblue')

  plt.legend(['Lampe $P_{CO2}=0$ mmHg', 'Lampe $P_{CO2}=10$ mmHg', 'Lampe $P_{CO2}=20$ mmHg', 'Lampe $P_{CO2}=30$ mmHg', 'Lampe $P_{CO2}=40$ mmHg',
              'Ours $P_{CO2}=0$ mmHg', 'Ours $P_{CO2}=10$ mmHg', 'Ours $P_{CO2}=20$ mmHg', 'Ours $P_{CO2}=30$ mmHg', 'Ours $P_{CO2}=40$ mmHg'])
  plt.ylabel("CBF (mL/s)")
  plt.xlabel("Arterial Pressure (mmHg)")
  plt.xlim(xmin=0, xmax=200)
  plt.show()

  return rmse, r_squared

if __name__ == "__main__":

  # State Variables
  C_a_initial = 0.15    # [mL/mmHg]
  P_ic_initial = 9.5    # [mmHg]
  initial_states = [C_a_initial, P_ic_initial]
  sim_time = 300        # s

  #####SIM 1!!!###
  P_a_list= []
  CBF_list = []
  for i in range(5):
    P_CO2 = P_CO2_sims[i]
    P_a, CBF = simulation1(C_a_initial, P_ic_initial, sim_time, P_CO2,1)
    P_a_list.append(P_a)
    CBF_list.append(CBF)

  CBF_lampe_0 = pd.read_csv('0.csv', header=None)
  cbf_0_Pa = CBF_lampe_0.iloc[:, 0].to_numpy()
  cbfs_0 = CBF_lampe_0.iloc[:, 1].to_numpy()

  plt.figure()
  plt.plot(P_a_list[0], CBF_list[0], 'r')
  plt.plot(P_a_list[1], CBF_list[1], '#5CB200')
  plt.plot(P_a_list[2], CBF_list[2], 'b')
  plt.plot(P_a_list[3], CBF_list[3], 'orchid')
  plt.plot(P_a_list[4], CBF_list[4], 'skyblue')
  plt.plot(cbf_0_Pa, np.zeros(cbf_0_Pa.size), marker=".")
  plt.xlim(xmin=0, xmax=200)
  plt.ylabel('CBF (mL/s)')
  plt.xlabel('Arterial Pressure (mmHg)')
  plt.legend(('$P_{CO2}=0$ mmHg', '$P_{CO2}=10$ mmHg', '$P_{CO2}=20$ mmHg', '$P_{CO2}=30$ mmHg', '$P_{CO2}=40$ mmHg', "lampe pa"), loc='upper left')
  plt.axhline(y=0, color='k')
  plt.axvline(x=0, color='k')
  # plt.show()

  ####### SIM 2!!! #######
  sim2_C_a = []
  sim2_P_ic = []
  sim2_P_a = []
  sim2_CBFs = []
  for P_vs in P_vs_sims:
    C_a_vals, P_ic_vals, P_a_vals, CBFs = simulation2(C_a_initial, P_ic_initial, P_vs, sim_time)
    sim2_C_a.append(C_a_vals)
    sim2_P_ic.append(P_ic_vals)
    sim2_P_a.append(P_a_vals)
    sim2_CBFs.append(CBFs)

  plt.figure()
  plt.plot(sim2_P_a[0], sim2_CBFs[0],'r')
  plt.plot(sim2_P_a[1], sim2_CBFs[1],'#5CB200')
  plt.plot(sim2_P_a[2], sim2_CBFs[2],'b')
  plt.plot(sim2_P_a[3], sim2_CBFs[3],'orchid')
  plt.plot(sim2_P_a[4], sim2_CBFs[4],'skyblue')
  plt.legend(["$P_{vs}=5$ mmHg", "$P_{vs}=10$ mmHg", "$P_{vs}=15$ mmHg", "$P_{vs}=25$ mmHg", "$P_{vs}=35$ mmHg"])
  plt.ylabel("CBF (mL/s)")
  plt.xlabel("Arterial Pressure (mmHg)")
  plt.xlim(xmin=0, xmax=200)
  # plt.show()

  fig, ax1 = plt.subplots()
  ax1.plot(sim2_P_a[0], sim2_CBFs[0],'r')
  ax1.plot(sim2_P_a[1], sim2_CBFs[1],'#5CB200')
  ax1.plot(sim2_P_a[2], sim2_CBFs[2],'b')
  ax1.plot(sim2_P_a[3], sim2_CBFs[3],'orchid')
  ax1.plot(sim2_P_a[4], sim2_CBFs[4],'skyblue')
  ax1.legend(["$P_{vs}=5$ mmHg", "$P_{vs}=10$ mmHg", "$P_{vs}=15$ mmHg", "$P_{vs}=25$ mmHg", "$P_{vs}=35$ mmHg"])
  ax1.set_ylabel("CBF (mL/s)")
  ax1.set_xlabel("Arterial Pressure (mmHg)")
  ax1.set_xlim(xmin=0, xmax=200)
  ax1.set_ylim(ymin=0, ymax=100)

  ax2 = plt.axes([0,0,1,1])
  ip = InsetPosition(ax1,[0.5,0.5,0.47,0.48])
  ax2.set_axes_locator(ip)
  mark_inset(ax1, ax2, loc1=2, loc2=4, fc='none', ec='0.5')
  
  ax2.plot(sim2_P_a[0], sim2_CBFs[0],'r')
  ax2.plot(sim2_P_a[1], sim2_CBFs[1],'#5CB200')
  ax2.plot(sim2_P_a[2], sim2_CBFs[2],'b')
  ax2.plot(sim2_P_a[3], sim2_CBFs[3],'orchid')
  ax2.plot(sim2_P_a[4], sim2_CBFs[4],'skyblue')
  ax2.set_ylim(ymin=15, ymax=15.1)
  ax2.set_xlim(xmin=78, xmax=78.2)
  ax2.set_xticks(np.arange(78, 78.2, 0.1))
  ax2.set_yticks(np.arange(15, 15.11, 0.05))

  #plt.show()

  ###### SIM #3!!###
  P_a_list_3 = []
  CBF_list_3 = []
  for i in range(5):
    P_CO2 = P_CO2_sims[i]
    P_a, CBF = simulation1(C_a_initial, P_ic_initial, sim_time, P_CO2, 3)
    P_a_list_3.append(P_a)
    CBF_list_3.append(CBF)
  plt.figure()
  plt.plot(P_a_list_3[0], CBF_list_3[0], 'r')
  plt.plot(P_a_list_3[1], CBF_list_3[1], '#5CB200')
  plt.plot(P_a_list_3[2], CBF_list_3[2], 'b')
  plt.plot(P_a_list_3[3], CBF_list_3[3], 'orchid')
  plt.plot(P_a_list_3[4], CBF_list_3[4], 'skyblue')
  plt.xlim(xmin=0, xmax=200)
  plt.ylabel('CBF (mL/s)')
  plt.xlabel('Arterial Pressure (mmHg)')
  plt.legend(('$P_{CO2}=0$ mmHg', '$P_{CO2}=10$ mmHg', '$P_{CO2}=20$ mmHg', '$P_{CO2}=30$ mmHg', '$P_{CO2}=40$ mmHg'), loc='upper left')
  plt.show()

  ######### difference between sim 1 and sim 3 #################
  print("RMSE sim 1 vs sim 3")
  norm = max(np.max(CBF_list[0]), np.max(CBF_list_3[0]))
  rmse_diff_0 = mean_squared_error(CBF_list[0], CBF_list_3[0],squared=False)  # y_true (lampes), #y_pred (ours)
  print(rmse_diff_0)

  norm = max(np.max(CBF_list[1]), np.max(CBF_list_3[1]))
  rmse_diff_10 = mean_squared_error(CBF_list[1], CBF_list_3[1], squared=False)  # y_true (lampes), #y_pred (ours)
  print(rmse_diff_10)

  norm = max(np.max(CBF_list[2]), np.max(CBF_list_3[2]))
  rmse_diff_20 = mean_squared_error(CBF_list[2], CBF_list_3[2], squared=False)  # y_true (lampes), #y_pred (ours)
  print(rmse_diff_20)

  norm = max(np.max(CBF_list[3]), np.max(CBF_list_3[3]))
  rmse_diff_30 = mean_squared_error(CBF_list[3], CBF_list_3[3], squared=False)  # y_true (lampes), #y_pred (ours)
  print(rmse_diff_30)

  norm = max(np.max(CBF_list[4]), np.max(CBF_list_3[4]))
  rmse_diff_40 = mean_squared_error(CBF_list[4], CBF_list_3[4], squared=False)  # y_true (lampes), #y_pred (ours)
  print(rmse_diff_40)

  print("R_SQUARED: sim 1 vs sim 3")
  rsquared_0 = r2_score(CBF_list[0], CBF_list_3[0]) #y_true, y_pred
  rsquared_10 = r2_score(CBF_list[1], CBF_list_3[1])
  rsquared_20 = r2_score(CBF_list[2], CBF_list_3[2])
  rsquared_30 = r2_score(CBF_list[3], CBF_list_3[3])
  rsquared_40 = r2_score(CBF_list[4], CBF_list_3[4])
  print(rsquared_0)
  print(rsquared_10)
  print(rsquared_20)
  print(rsquared_30)
  print(rsquared_40)



  ##### error our sim 1 vs lampe's results
  rmse, r_squared = fig_9_lampe()
  print("RMSE: our sim 1 vs lampes")
  print(rmse[0])
  print(rmse[1])
  print(rmse[2])
  print(rmse[3])
  print(rmse[4])

  print("R_SQUARED: our sim 1 vs lampes")
  print(r_squared[0])
  print(r_squared[1])
  print(r_squared[2])
  print(r_squared[3])
  print(r_squared[4])

