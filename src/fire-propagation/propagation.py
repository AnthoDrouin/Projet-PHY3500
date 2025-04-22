import numpy as np
from typing import List, Tuple, Dict, Any
from parameters import Parameters
import matplotlib.pyplot as plt
from tqdm import tqdm


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
		self.scalars = {}
		self.misc = {
			"current_time": 0,
		}

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

		assert self.params.theta == 0, "theta must be 0 for now, WILL BE IMPLEMENTED LATER"

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
		# Gaussian initial condition (any grid size)

		#self.grid["temp"] = self.gaussian(
		# 	x=self.x,
		# 	y=self.y,
		# 	x0=self.position_max_temp_initial[0],
		# 	y0=self.position_max_temp_initial[1],
		# 	sigma=self.sigma,
		# 	temp_max=self.temperature_max_initial_condition,
		# 	temp_amb=self.params.ambiant_temperature
		#)

		# Initial condition as a rectangle (GRID 200X200 ONLY!!!)

		self.grid["temp"] = np.zeros(self.misc["dim_grid"]) + self.params.ambiant_temperature
		height = 30
		width = 10
		center_y, center_x = 200, 50
		start_y = center_y - height // 2
		end_y = center_y + height // 2
		start_x = center_x - width // 2
		end_x = center_x + width // 2
		self.grid["temp"][start_y:end_y, start_x:end_x] = self.temperature_max_initial_condition

		sigma_y = height / 2
		sigma_x = width / 2
		y_range = np.arange(start_y, end_y)
		x_range = np.arange(start_x, end_x)
		Y, X = np.meshgrid(y_range, x_range, indexing="ij")
		gaussian = np.exp(-(((Y - center_y) ** 2) / (2 * sigma_y ** 2) + ((X - center_x) ** 2) / (2 * sigma_x ** 2)))
		T_max = self.temperature_max_initial_condition
		T_amb = self.params.ambiant_temperature
		smoothed_patch = T_amb + (T_max - T_amb) * gaussian
		self.grid["temp"][start_y:end_y, start_x:end_x] = smoothed_patch

	def update_dispersion_grid(self):
		"""
		Initial conditions for Deffx and Deffy
		"""
		l_x, l_y = self.width_burning_zone
		#threshold = 0.1 * np.max(self.grid["temp"]) + self.params.ambiant_temperature
		#idx_closest = np.unravel_index(np.argmin(np.abs(self.grid["temp"] - threshold)), self.grid["temp"].shape)
		# TODO: The computation of l_x and l_y might needs to be complexified
		# TODO: MODIFICATION HERE -> ADD ABSOLUTE VALUE
		#l_x = np.abs(self.x[idx_closest[0], idx_closest[1]])
		#l_y = np.abs(self.y[idx_closest[0], idx_closest[1]])
		#if self.misc["current_time"] == 0.0:
		#	norm_l = np.sqrt(l_x ** 2 + l_y ** 2)
		#	l_x = norm_l / np.sqrt(2)
		#	l_y = norm_l / np.sqrt(2)
		# Initially a gaussian, so it is expected that L_x = L_y

		w_x, w_y = self.fire_width

		if self.misc["current_time"] == 0.0:
			# It is initially a gaussian, so it is expected that W_x = W_y
			w_norm = np.sqrt(w_x ** 2 + w_y ** 2)
			w_x = w_norm / np.sqrt(2)
			w_y = w_norm / np.sqrt(2)

		self.scalars["Deffx"] = self.params.d_rb + self.params.a_d * self.params.avg_canopy_velocity[0] * l_x * (
				1 - np.exp(-self.params.gamma_d * w_x))
		self.scalars["Deffy"] = self.params.d_rb + self.params.a_d * self.params.avg_canopy_velocity[1] * l_y * (
				1 - np.exp(-self.params.gamma_d * w_y))

	@property
	def width_burning_zone(self):
		threshold = 0.1 * np.max(self.grid["temp"]) + self.params.ambiant_temperature
		idx_threshold_temp = np.unravel_index(np.argmin(np.abs(self.grid["temp"] - threshold)), self.grid["temp"].shape)
		#test = np.abs(self.grid["temp"] - threshold)
		#idx_threshold_temp = np.unravel_index(np.where(np.abs(self.grid["temp"] - threshold) < tol), self.grid["temp"].shape)
		x_threshold_temp = self.x[idx_threshold_temp]
		y_threshold_temp = self.y[idx_threshold_temp]

		idx_max_temp = np.unravel_index(np.argmax(self.grid["temp"]), self.grid["temp"].shape)
		x_max_temp = self.x[idx_max_temp]
		y_max_temp = self.y[idx_max_temp]

		l_x = np.abs(x_max_temp - x_threshold_temp)
		l_y = np.abs(y_max_temp - y_threshold_temp)

		return l_x, l_y

	def update_advection_grid(self):
		"""
		Initial condition for <u_effx>, <u_effy>
		"""
		# TODO: IMPLEMENT u_buoy -> FOR NOW THETA=0 (else u_buoy needs to be computed...)
		# I don't want/know how to implement I_B especially the rate of spread ROS

		x_c = self.grid["s_2"] / self.scalars["s_2_0"]
		self.grid["<u_x>"] = self.params.avg_canopy_velocity[0] + (
					self.params.avg_velocity_bare_ground[0] - self.params.avg_canopy_velocity[0]) * (
					                     1 - x_c)  # Eq20 with S_2 = S_2_0 -> x_c = 1
		self.grid["<u_y>"] = self.params.avg_canopy_velocity[1] + (
					self.params.avg_velocity_bare_ground[1] - self.params.avg_canopy_velocity[1]) * (
					                     1 - x_c)  # Eq20 with S_2 = S_2_0 -> x_c = 1
		self.grid["<u_effx>"] = np.sqrt(
			(self.grid["<u_x>"] * np.sin(self.params.psi)) ** 2 + (self.grid["<u_x>"] * np.cos(self.params.psi)) ** 2)
		self.grid["<u_effy>"] = np.sqrt(
			(self.grid["<u_y>"] * np.sin(self.params.psi)) ** 2 + (self.grid["<u_y>"] * np.cos(self.params.psi)) ** 2)

	def initial_conditions_reaction_grid(self):
		"""
		Initial parameters for the reaction grod, it includes:
		m_s_0, m_s_1_0, m_s_2_0 and s_2_0
		"""
		self.scalars["m_s_0"] = self.params.alpha * self.params.rho_solid
		self.scalars["m_s_2_0"] = (self.params.alpha * self.params.rho_solid) / ((self.params.fmc / 100) + 1)
		self.scalars["m_s_1_0"] = (self.params.fmc / 100) * self.scalars["m_s_2_0"]

		self.scalars["s_2_0"] = self.scalars["m_s_2_0"] / self.scalars["m_s_0"]

		self.grid["s_1"] = np.ones(self.misc["dim_grid"]) * (self.scalars["m_s_1_0"] / self.scalars["m_s_0"])
		self.grid["s_2"] = np.ones(self.misc["dim_grid"]) * (self.scalars["m_s_2_0"] / self.scalars["m_s_0"])

	def update_reaction_grid(self):
		"""
		Initial conditions for the reaction grid. This implies the computation of the following parameters:
		S, S1, S2, m_s, m_s1, m_s2, m_g, c0, c1
		"""
		# Compute S, S1, S2

		r_1 = self.params.cs1 * np.exp(-self.params.b1 / self.grid["temp"])
		s_1 = np.exp(-r_1 * self.misc["current_time"]) * (self.scalars["m_s_1_0"] / (self.params.alpha * self.params.rho_solid))
		#s_1 = np.exp(-r_1 * self.integration_step) * self.grid["s_1"]
		r_2 = self.params.cs2 * np.exp(-self.params.b2 / self.grid["temp"])
		#avg_velocity_through_canopy = np.sqrt((self.params.avg_canopy_velocity[0] ** 2) + (self.params.avg_canopy_velocity[1] ** 2))
		#r_m = self.params.r_m_0 + (self.params.r_m_c * (avg_velocity_through_canopy - 1))
		r_m = 1e-2

		#r_m = 6e-3
		#r_m = 1*self.grid["temp"].max()

		r_2t = (r_2 * r_m) / (r_2 + r_m)
		s_2 = np.exp(-r_2t * self.misc["current_time"]) * (self.scalars["m_s_2_0"] / (self.params.alpha * self.params.rho_solid))
		#s_2 = np.exp(-r_2t * self.integration_step) * self.grid["s_2"]

		s = s_1 + s_2

		if self.misc["current_time"] == 0.0:
			self.grid["s_1"] = s_1
			self.grid["s_2"] = s_2
			self.grid["s"] = s
		else:
			self.grid["s_1"] = np.minimum(self.grid["s_1"], s_1)
			self.grid["s_2"] = np.minimum(self.grid["s_2"], s_2)
			self.grid["s"] = self.grid["s_1"] + self.grid["s_2"]

		#self.grid["s_1"] = s_1
		#self.grid["s_2"] = s_2
		#self.grid["s"] = s

		self.grid["r_1"] = r_1
		self.grid["r_2t"] = r_2t
		# Compute c0, c1

		c0 = self.params.alpha * s + ((1 - self.params.alpha) * self.params.lambda_ * self.params.gamma) + (
					self.params.alpha * self.params.gamma * (1 - s))
		c1 = c0 - self.params.alpha * s

		self.grid["c_0"] = c0
		self.grid["c_1"] = c1

		# Compute, m_s, m_s1, m_s2, m_g

		self.grid["m_s_1"] = s_1 * self.scalars["m_s_0"]
		self.grid["m_s_2"] = s_2 * self.scalars["m_s_0"]
		self.grid["m_s"] = self.grid["m_s_1"] + self.grid["m_s_2"]

	def update_convection(self):
		"""
		Initial conditions for U(T-Ta)
		"""
		self.grid["U"] = (self.params.a_nc * (np.cbrt(self.grid["temp"] - self.params.ambiant_temperature))) + (
				self.params.epsilon * self.sigma_b * (
				(self.grid["temp"] ** 2) + self.params.ambiant_temperature ** 2) * (
						self.grid["temp"] + self.params.ambiant_temperature))

	def setup(self):
		self.prepare_grid()
		self.initial_condition_temp_grid()
		self.initial_conditions_reaction_grid()
		self.update()

	def run(self):
		self.setup()

		prev_step = 0
		for t in tqdm(self.time):
			if t == 0:
				d_temp_over_d_time = self.d_temp_over_d_time()
				current_temperature = self.grid["temp"] + self.integration_step * d_temp_over_d_time
				prev_step = d_temp_over_d_time
			else:
				d_temp_over_d_time = self.d_temp_over_d_time()
				current_temperature = self.grid["temp"] + (self.integration_step/2) * (3 * d_temp_over_d_time - prev_step)
				prev_step = d_temp_over_d_time

			self.grid["temp"] = current_temperature
			self.misc["current_time"] = t
			print(self.grid["temp"].max())
			if current_temperature.max() < 575:
				break

			self.update()

	def d_temp_over_d_time(self):
		"""
		Compute dT/dt with the current parameters
		:return: dT/dt
		"""
		# DISPERSION #
		dispersion = self.scalars["Deffx"] * self.d2_temp_over_dx2 + self.scalars["Deffy"] * self.d2_temp_over_dy2

		# ADVECTION #
		advection = -self.grid["<u_effx>"] * self.d_temp_over_dx - self.grid["<u_effy>"] * self.d_temp_over_dy

		# REACTION #
		reaction = -self.params.c2 * self.grid["s_1"] * self.grid["r_1"] + self.params.c3 * self.grid["s_2"] * self.grid["r_2t"]

		# CONVECTION #
		convection = -self.params.c4 * self.grid["U"]
		d_temp_dt = self.grid["c_1"] * (dispersion + advection) + reaction + convection

		return d_temp_dt / self.grid["c_0"]

	def border_conditions(self):
		self.grid["temp"][self.grid["temp"] < self.params.ambiant_temperature] = self.params.ambiant_temperature

	def update(self):
		self.update_reaction_grid()
		self.update_advection_grid()
		self.update_dispersion_grid()
		self.update_convection()
		self.border_conditions()

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

	@property
	def d_temp_over_dx(self):
		"""
		Partial derivative of the temperature with respect to x.
		"""
		return np.gradient(self.grid["temp"], self.spacing[0], axis=1)

	@property
	def d2_temp_over_dx2(self):
		return np.gradient(self.d_temp_over_dx, self.spacing[0], axis=1)

	@property
	def d_temp_over_dy(self):
		"""
		Partial derivative of the temperature with respect to y.
		"""
		return np.gradient(self.grid["temp"], self.spacing[1], axis=0)

	@property
	def d2_temp_over_dy2(self):
		return np.gradient(self.d_temp_over_dy, self.spacing[1], axis=0)

	@property
	def fire_width(self):
		idx_max_temp_x = np.argmax(self.grid["temp"], axis=0)
		# Max temperature in each column (for each x) -> Return the index of the ROW

		max_temp = np.max(self.grid["temp"], axis=0)
		max_temp_x = self.x[0, :]
		max_temp_y = self.y[:, 0][idx_max_temp_x]

		max_temp_x_550 = max_temp_x[np.where(max_temp > 550)]
		max_temp_y_550 = max_temp_y[np.where(max_temp > 550)]

		max_temp_pts = np.array([max_temp_x_550, max_temp_y_550])
		distances = np.linalg.norm(max_temp_pts[:, :, None] - max_temp_pts[:, None, :], axis=0)
		max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
		max_temp_pt1 = max_temp_pts[:, max_distance_idx[0]]
		max_temp_pt2 = max_temp_pts[:, max_distance_idx[1]]

		w_x = np.abs(max_temp_pt1[0] - max_temp_pt2[0])
		w_y = np.abs(max_temp_pt1[1] - max_temp_pt2[1])

		return w_x, w_y

	@property
	def sigma_b(self):
		"""
		:return: Stefan-Boltzmann constant
		"""
		return 5.670374419e-8


if __name__ == "__main__":
	pass
