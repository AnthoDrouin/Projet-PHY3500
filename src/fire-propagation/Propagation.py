import numpy as np
from typing import List, Tuple, Dict, Any
from parameters import Parameters


class Propagation:

	def __init__(
			self,
			parameters: Parameters,
			grid_size: Tuple[int, int],
			spacing: Tuple[float, float],
			integration_time: float,
			integration_step: float,
			**kwargs: Dict[str, Any]
	):
		self.parameters = parameters
		self.grid_size = grid_size
		self.spacing = spacing
		self.integration_time = integration_time
		self.integration_step = integration_step

	def check_params(self):
		pass

	def initial_coefficients(self):
		pass

	def initial_conditions(self):
		pass

	def setup(self):
		pass

	def run(self):
		pass

	def get_results(self):
		pass
