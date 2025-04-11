import numpy as np
from typing import *


class Parameters:

	def __init__(
			self,
			u10: List[float],
			z0: float,
			delta: float,
			fmc: float = 25,
			height_canopy: float = 2,
			ambiant_temperature: float = 300,
			rho_gas: float = 1,
			rho_solid: float = 700,
			heat_capacity_gas: float = 1043,
			heat_capacity_solid: float = 1800,
			cs1: float = 30,
			cs2: float = 40,
			b1: float = 4500,
			b2: float = 7000,
			a1: float = 22e5,
			a2: float = 2e7,
			d_rb: float = 0.1,
			r_m_0: float = 0.002,
			r_m_c: float = 0.004,
			gamma_d: float = 0.03,
			a_nc: float = 0.2,
			a_d: float = 0.125,
			eta: float = 3,
			alpha: float = 0.002,
			epsilon: float = 0.2,
			sigma: float = 20.0,
			**kwargs: Dict[str, Any]
	):
		self.u10 = u10
		self.z0 = z0
		self.delta = delta
		self.fmc = fmc
		self.height_canopy = height_canopy
		self.ambiant_temperature = ambiant_temperature
		self.rho_gas = rho_gas
		self.rho_solid = rho_solid
		self.heat_capacity_gas = heat_capacity_gas
		self.heat_capacity_solid = heat_capacity_solid
		self.cs1 = cs1
		self.cs2 = cs2
		self.b1 = b1
		self.b2 = b2
		self.a1 = a1
		self.a2 = a2
		self.d_rb = d_rb
		self.r_m_0 = r_m_0
		self.r_m_c = r_m_c
		self.gamma_d = gamma_d
		self.a_nc = a_nc
		self.a_d = a_d
		self.eta = eta
		self.alpha = alpha
		self.epsilon = epsilon
		self.sigma = sigma

		self.kwargs = kwargs

		self.gamma = None
		self.lambda_ = None
		self.c0, self.c1, self.c2, self.c3, self.c4 = None, None, None, None, None
		self.avg_canopy_velocity = [None, None]

		self.compute_constant_params()

	def compute_constant_params(self):
		self._compute_gamma()
		self._compute_lambda()
		self._compute_coefficients()
		self._compute_average_canopy_velocity()

	def _compute_gamma(self):
		self.gamma = self.heat_capacity_gas / self.heat_capacity_solid

	def _compute_lambda(self):
		self.lambda_ = self.rho_gas / self.rho_solid

	def _compute_coefficients(self):
		# Fix coefficients
		self.c2 = self.alpha * self.a1 / self.heat_capacity_solid
		self.c3 = self.alpha * self.a2 / self.heat_capacity_solid
		self.c4 = 1 / (self.height_canopy * self.rho_solid * self.heat_capacity_solid)

	def _compute_average_canopy_velocity(self):
		kappa = 0.41
		u10_x, u10_y = self.u10

		d_x = (self.height_canopy - self.z0) - (self.delta * u10_x)
		d_y = (self.height_canopy - self.z0) - (self.delta * u10_y)

		friction_velocity_x = u10_x * kappa / np.log(10 * d_x / self.z0)
		friction_velocity_y = u10_y * kappa / np.log(10 * d_y / self.z0)

		# velocity at the top of the canopy
		u_h_x = (friction_velocity_x / kappa) * np.log((self.height_canopy - d_x) / self.z0)
		u_h_y = (friction_velocity_y / kappa) * np.log((self.height_canopy - d_y) / self.z0)

		avg_canopy_velocity_x = (u_h_x / self.eta) * (1 - np.exp(-self.eta))
		avg_canopy_velocity_y = (u_h_y / self.eta) * (1 - np.exp(-self.eta))
		self.avg_canopy_velocity = [avg_canopy_velocity_x, avg_canopy_velocity_y]


if __name__ == "__main__":
	# Example usage
	params = Parameters(u10=[10.0, 0.0], z0=0.5, delta=0.08)
	print("Average Canopy Velocity:", params.avg_canopy_velocity)
	print("Gamma:", params.gamma)
	print("Lambda:", params.lambda_)
	print("C0:", params.c0)
	print("C1:", params.c1)
	print("C2:", params.c2)
	print("C3:", params.c3)
	print("C4:", params.c4)
