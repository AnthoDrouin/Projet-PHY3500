import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
from propagation import Propagation


parameters = Parameters(u10=[10.0, 0.0], z0=0.5, delta=0.08)
grid_size = (500, 500)
spacing = (0.5, 0.5)
integration_time = 100.0
integration_step = 0.1

propagation = Propagation(
	parameters,
	grid_size,
	spacing,
	integration_time,
	integration_step,

)

temp_matrix = propagation.grid["temp"]

x = propagation.x
y = propagation.y
plt.imshow(temp_matrix, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='hot')
plt.colorbar()
plt.show()