import numpy as np
from typing import List, Tuple, Dict, Any
from parameters import Parameters
import matplotlib.pyplot as plt


class Propagation:

	def __init__(
			self,
			parameters: Parameters,
			grid_size: Tuple[int, int],
			spacing: Tuple[float, float],
			integration_time: float,
			integration_step: float,
			temperature_max_initial_condition: float = 1200,
			position_max_temp_initial: Tuple[float, float] = (0, 0),
			sigma: float = 20.0,
			**kwargs: Dict[str, Any]
	):
		self.params = parameters
		self.grid_size = grid_size
		self.spacing = spacing
		self.integration_time = integration_time
		self.integration_step = integration_step

		self.temperature_max_initial_condition = temperature_max_initial_condition
		self.position_max_temp_initial = position_max_temp_initial
		self.sigma = sigma

		self.check_params()

		self.time = np.linspace(0, self.integration_time, int(self.integration_time / self.integration_step))
		self.num_time_step = int(self.integration_time / self.integration_step)

		self.x, self.y = None, None
		self.grid = {}
		self.misc = {}

		self.run()

	def check_params(self):
		assert isinstance(self.grid_size, tuple), "grid_size must be a tuple"
		assert len(self.grid_size) == 2, "grid_size must be a tuple of length 2"
		assert isinstance(self.spacing, tuple), "spacing must be a tuple"
		assert len(self.spacing) == 2, "spacing must be a tuple of length 2"
		assert isinstance(self.integration_time, (int, float)), "integration_time must be a number"
		assert isinstance(self.integration_step, (int, float)), "integration_step must be a number"
		assert self.integration_step > 0, "integration_step must be positive"
		assert self.integration_time > 0, "integration_time must be positive"
		assert self.integration_time > self.integration_step, "integration_time must be greater than integration_step"
		assert self.grid_size[0] > 0, "grid_size[0] must be positive"
		assert self.grid_size[1] > 0, "grid_size[1] must be positive"
		assert self.spacing[0] > 0, "spacing[0] must be positive"
		assert self.spacing[1] > 0, "spacing[1] must be positive"

	def prepare_grid(self):
		n = int(self.grid_size[0] / self.spacing[0])
		x = np.linspace(0, self.grid_size[0] - self.spacing[0], n)
		y = np.linspace(0, self.grid_size[1] - self.spacing[1], n)
		self.x, self.y = np.meshgrid(x, y)
		self.x = self.x - self.grid_size[0] / 2  # [-x, x]
		self.y = -self.y + self.grid_size[1] / 2  # [-y, y]

		dim_grid = (n, n)
		self.misc["dim_grid"] = dim_grid

	def initial_condition_temp_grid(self):
		"""
		Initial conditions for the temperature grid.
		"""
		self.grid["temp"] = self.gaussian(
			x=self.x,
			y=self.y,
			x0=self.position_max_temp_initial[0],
			y0=self.position_max_temp_initial[1],
			sigma=self.sigma,
			temp_max=self.temperature_max_initial_condition,
			temp_amb=self.params.ambiant_temperature
		)

	def initial_conditions_dispersion_grid(self):
		pass

	def initial_conditions_advection_grid(self):
		"""
		Initial condition for <u_effx>, <u_effy>, dT/dx, dT/dy
		"""
		pass

	def initial_conditions_reaction_grid(self):
		"""
		Initial conditions for the reaction grid. This implies the computation of the following parameters:
		S, S1, S2, m_s, m_s1, m_s2, m_g, c0, c1
		"""
		m_s_2_0 = (self.params.alpha * self.params.rho_solid) / ((self.params.fmc / 100) + 1)
		m_s_1_0 = (self.params.fmc / 100) * m_s_2_0
		m_s_0 = self.params.alpha * self.params.rho_solid
		m_g = self.params.alpha*self.params.rho_solid + (1 - self.params.alpha) * self.params.rho_gas - m_s_0

		s_1_0 = m_s_1_0 / (self.params.alpha * self.params.rho_solid)
		s_2_0 = m_s_2_0 / (self.params.alpha * self.params.rho_solid)
		s_0 = s_1_0 + s_2_0

		c0 = self.params.alpha * s_0 + (
				1 - self.params.alpha) * self.params.lambda_ * self.params.gamma + self.params.alpha * self.params.gamma * (1 - s_0)
		c1 = c0 - self.params.alpha * s_0

		# Setup grid

		self.grid["c_0"] = np.ones(self.misc["dim_grid"]) * c0
		self.grid["c_1"] = np.ones(self.misc["dim_grid"]) * c1
		self.grid["s_1"] = np.ones(self.misc["dim_grid"]) * s_1_0
		self.grid["s_2"] = np.ones(self.misc["dim_grid"]) * s_2_0
		self.grid["s"] = np.ones(self.misc["dim_grid"]) * s_0
		self.grid["m_s"] = np.ones(self.misc["dim_grid"]) * m_s_0
		self.grid["m_s_1"] = np.ones(self.misc["dim_grid"]) * m_s_1_0
		self.grid["m_s_2"] = np.ones(self.misc["dim_grid"]) * m_s_2_0
		self.grid["m_g"] = np.ones(self.misc["dim_grid"]) * m_g

	def initial_conditions_convection(self):
		pass

	def setup(self):
		self.prepare_grid()
		self.initial_condition_temp_grid()
		self.initial_conditions_reaction_grid()

	def run(self):
		self.setup()
		pass

	def tracker(self):
		pass

	def get_results(self):
		pass

	@staticmethod
	def gaussian(x: float, y: float, x0: float, y0: float, sigma: float, temp_max: float, temp_amb) -> float:
		"""
		Calculate the Gaussian function for a given point (x, y) with respect to a center point (x0, y0) and standard deviation sigma.
		:param x: Position in the x direction
		:param y: Position in the y direction
		:param x0: Position of the center in the x direction
		:param y0: Position of the center in the y direction
		:param sigma: Standard deviation of the Gaussian function
		:param temp_max: Maximum temperature at the center of the Gaussian
		:param temp_amb: Ambient temperature
		:return: temp at the point (x, y)
		"""
		return temp_amb + (temp_max - temp_amb) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


if __name__ == "__main__":
	pass
