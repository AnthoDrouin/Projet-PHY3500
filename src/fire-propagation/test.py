import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
from propagation import Propagation
from matplotlib.colors import LinearSegmentedColormap

print("Processing data RK4")
data_rk4 = np.load("RK4_temp_max_temp_grid_time_exec_time_u10_3_rm_9_65_001_01.npy", allow_pickle=True).item()
print("Processing data RK2")
data_rk2 = np.load("RK2_temp_max_temp_grid_time_exec_time_u10_3_rm_9_65_001_01.npy", allow_pickle=True).item()
print("Processing data AM")
data_am = np.load("AM_temp_max_temp_grid_time_exec_time_u10_3_rm_9_65_01.npy", allow_pickle=True).item()
print("Processing data Euler")
data_euler = np.load("Euler_temp_max_temp_grid_time_exec_time_u10_3_rm_9_65_01.npy", allow_pickle=True).item()
print("Processing data Leapfrog")
data_leapfrog = np.load("Leapfrog_temp_max_temp_grid_time_exec_time_u10_3_rm_9_65_01.npy", allow_pickle=True).item()

plt.style.use("https://raw.githubusercontent.com/dccote/Enseignement/master/SRC/dccote-errorbars.mplstyle")

plt.figure(figsize=(16.2, 10))
plt.plot(data_rk4["time"], data_rk4["temp_max"], "-", label="RK4", color="blue")
plt.plot(data_rk2["time"], data_rk2["temp_max"], "-", label="RK2", color="orange")
plt.plot(data_am["time"], data_am["temp_max"], "-", label="AM", color="green")
plt.plot(data_euler["time"], data_euler["temp_max"], "-", label="Euler", color="red")
plt.plot(data_leapfrog["time"][0:np.argmax(data_leapfrog["temp_max"])], data_leapfrog["temp_max"][0:np.argmax(data_leapfrog["temp_max"])], "-", label="Saute-Mouton", color="purple")
plt.xlabel("Temps [s]", fontsize=22)
plt.ylabel("Température maximale [K]", fontsize=22)
plt.tick_params(axis="both", which="major", labelsize=22)
plt.legend(frameon=False, fontsize=22)
plt.show()

plt.figure(figsize=(16.2, 10))
plt.plot(data_rk4["time"], data_rk4["temp_max"], "-", label="RK4", color="blue")
plt.plot(data_rk2["time"], data_rk2["temp_max"], "-", label="RK2", color="orange")
plt.plot(data_am["time"], data_am["temp_max"], "-", label="AM", color="green")
plt.plot(data_euler["time"], data_euler["temp_max"], "-", label="Euler", color="red")
plt.xlabel("Temps [s]", fontsize=22)
plt.ylabel("Température maximale [K]", fontsize=22)
plt.tick_params(axis="both", which="major", labelsize=22)
plt.legend(frameon=False, fontsize=22)
plt.show()

diff_rk2 = data_rk2["temp_max"] - data_rk4["temp_max"]
diff_am = data_am["temp_max"] - data_rk4["temp_max"]
diff_euler = data_euler["temp_max"] - data_rk4["temp_max"]

plt.figure(figsize=(16.2, 10))

plt.plot(data_rk2["time"], diff_rk2, "-", label="RK2", color="orange")
plt.plot(data_am["time"], diff_am, "-", label="AM", color="green")
plt.plot(data_euler["time"], diff_euler, "-", label="Euler", color="red")
plt.xlabel("Temps [s]", fontsize=22)
plt.ylabel("Différence avec RK4 [K]", fontsize=22)
plt.tick_params(axis="both", which="major", labelsize=22)
plt.legend(frameon=False, fontsize=22)
plt.show()

print(f"RK4: {data_rk4['execution_time']}")
print(f"RK2: {data_rk2['execution_time']}")
print(f"AM: {data_am['execution_time']}")
print(f"Euler: {data_euler['execution_time']}")
print(f"Leapfrog: {data_leapfrog['execution_time']}")



exit()
parameters = Parameters(u10=[3/np.sqrt(2), 3/np.sqrt(2)], z0=0.5, delta=0.08)
grid_size = (200, 200)
spacing = (0.5, 0.5)
integration_time = 400
integration_step = 0.1

propagation = Propagation(
	parameters,
	grid_size,
	spacing,
	integration_time,
	integration_step,
	position_max_temp_initial=(-75, 0),
	temperature_max_initial_condition=800,
	sigma=5,
)

temp_matrix = propagation.grid["s"]

# print(propagation.grid["temp"])

x = propagation.x
y = propagation.y
plt.imshow(temp_matrix, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap="inferno", vmin=0, vmax=1)
plt.colorbar()
plt.show()


temp_matrix = propagation.grid["temp"]

# # print(propagation.grid["temp"])

x = propagation.x
y = propagation.y
plt.imshow(temp_matrix, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='hot')
plt.colorbar()
plt.show()

#temp_matrix = temp_matrix[500, :]

#print(temp_matrix.shape)

#plt.plot(x[500, :], temp_matrix)
#plt.show()