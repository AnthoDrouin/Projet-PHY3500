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
			psi: float = 0,
			theta: float = 0,
			**kwargs: Dict[str, Any]
	):
		"""
		:param u10: Wind speed at 10m above ground level (m/s) [u10_x, u10_y]
		:param z0: Surface roughness at the top of the canopy (m)
		:param delta: TODO
		:param fmc: Fuel moisture content (%)
		:param height_canopy: Height of the canopy (m)
		:param ambiant_temperature: Ambient temperature (K)
		:param rho_gas: Density of the gas (kg/m^3)
		:param rho_solid: Density of the solid (kg/m^3)
		:param heat_capacity_gas: Heat capacity of the gas (J/(kg*K))
		:param heat_capacity_solid: Heat capacity of the solid (J/(kg*K))
		:param cs1: Constant for r1 (s^-1)
		:param cs2: Constant for r2 (s^-1)
		:param b1: Quantify difference in behavior between dead and live moisture content (K)
		:param b2: Depend on fuel characteristics and determine the intensity of pyrolysis and combustion (K)
		:param a1: Standard heat of the endothermic (water vaporization) reaction (J/kg)
		:param a2: Standard heat of the exothermic (pyrolysis and combustion) reaction (J/kg)
		:param d_rb: Contribution of the radiation and buoyancy in the absence of wind (m^2/s)
		:param r_m_0: Curve fit parameter for r_m (rate of oxygen delivery)
		:param r_m_c: Curve fit parameter for r_m (rate of oxygen delivery)
		:param gamma_d: Increases or decreases the fireline length at which the asymptotic ROS is reached (m^-1)
		:param a_nc: Sums up terms for the correlation Nu_{nc} = 0.15Gr^{1/3}Pr^{1/3} valid for a horizontal hot surface (W/m^2K^(4/3))
		:param a_d: Constant for D_effx and D_effy
		:param eta: Structure of air flow within crop canopies constant
		:param alpha: Packing ratio of solids (solid m^3/total m^3)
		:param epsilon:
		:param psi: Angle between the wind direction and upslope direction
		:param theta: Inclination of the slope
		:param kwargs:
		"""
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
		self.psi = psi
		self.theta = theta

		self.kwargs = kwargs

		self.gamma = None
		self.lambda_ = None
		self.c0, self.c1, self.c2, self.c3, self.c4 = None, None, None, None, None
		self.avg_canopy_velocity = [None, None]
		self.avg_velocity_bare_ground = [None, None]

		self.compute_constant_params()

	def compute_constant_params(self):
		self._compute_gamma()
		self._compute_lambda()
		self._compute_coefficients()
		self._compute_average_canopy_velocity()
		self._compute_average_velocity_over_bare_ground()

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

		friction_velocity_x = u10_x * kappa / np.log((10 - d_x) / self.z0)
		friction_velocity_y = u10_y * kappa / np.log((10 - d_y) / self.z0)

		# velocity at the top of the canopy
		u_h_x = (friction_velocity_x / kappa) * np.log((self.height_canopy - d_x) / self.z0)
		u_h_y = (friction_velocity_y / kappa) * np.log((self.height_canopy - d_y) / self.z0)

		avg_canopy_velocity_x = (u_h_x / self.eta) * (1 - np.exp(-self.eta))
		avg_canopy_velocity_y = (u_h_y / self.eta) * (1 - np.exp(-self.eta))
		self.avg_canopy_velocity = [avg_canopy_velocity_x, avg_canopy_velocity_y]

	def _compute_average_velocity_over_bare_ground(self):
		kappa = 0.41
		u10_x, u10_y = self.u10

		friction_velocity_bare_ground_x = u10_x * kappa / np.log(10/self.z0)
		friction_velocity_bare_ground_y = u10_y * kappa / np.log(10/self.z0)

		avg_velocity_bare_ground_x = (friction_velocity_bare_ground_x / kappa) * ((self.height_canopy / (self.height_canopy - self.z0)) * np.log(self.height_canopy / self.z0) - 1)
		avg_velocity_bare_ground_y = (friction_velocity_bare_ground_y / kappa) * ((self.height_canopy / (self.height_canopy - self.z0)) * np.log(self.height_canopy / self.z0) - 1)

		self.avg_velocity_bare_ground = [avg_velocity_bare_ground_x, avg_velocity_bare_ground_y]

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
