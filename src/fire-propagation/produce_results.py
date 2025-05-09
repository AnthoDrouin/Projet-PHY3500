import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
from propagation import Propagation
from matplotlib.colors import LinearSegmentedColormap


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
