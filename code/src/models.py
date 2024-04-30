"""
TODO 
- Equivariant layer 
- invariant layer 
- Gauge equivariant NN 
"""

import flax.linen as nn 
import jax.numpy as np


class EquivariantCNN(nn.Module): 

	# compact lets you define all the layers IN call as opposed to in init like 
	# in tensorflow. I can imagine that for skip connections this might be bad. 
	@nn.compact 
	def __call__(self, x): 
		# TODO 
		pass 

class InvariantCNN(nn.Module): 

	@nn.compact 
	def __call__(self, x): 
		# TODO 
		pass 

class GENN(nn.Module): 
	@nn.compact 
	def __call__(self, x): 
		# TODO 
		pass 

