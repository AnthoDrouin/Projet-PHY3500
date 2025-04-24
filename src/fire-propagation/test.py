import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
from propagation import Propagation
from matplotlib.colors import LinearSegmentedColormap



parameters = Parameters(u10=[5,0], z0=0.25, delta=0.04)
grid_size = (200, 200)
spacing = (0.5, 0.5)
integration_time = 500
integration_step = .1

propagation = Propagation(
	parameters,
	grid_size,
	spacing,
	integration_time,
	integration_step,
	position_max_temp_initial=(-90, 0),
	temperature_max_initial_condition=800,
	sigma=10,
	save_path="src\\fire-propagation\\figures\\ux5_z25_d4_Ti800"
)

temp_matrix = propagation.grid["s_1"]

# print(propagation.grid["temp"])

x = propagation.x
y = propagation.y

plt.imshow(temp_matrix, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap="inferno", vmin=0, vmax=1)
plt.colorbar()
plt.show()



temp_matrix = propagation.grid["s_2"]

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