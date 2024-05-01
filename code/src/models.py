"""
TODO 
- Equivariant layer 
- invariant layer 
- Gauge equivariant NN 
*** ALL WEIGHTS SHOULD BE REAL VALUED ***
"""

import flax.linen as nn 
import jax.numpy as np
from wilson import * 


class EquivariantCNN(nn.Module): 
    # include parameters like hidden size here: 
    graph = None
    kernel_size = (3, 3)

    # compact lets you define all the layers IN call as opposed to in init like 
    # in tensorflow. I can imagine that for skip connections this might be bad. 
    @nn.compact 
    def __call__(self, x): 
        # Input should be of shape (B, N); B = batch size, N = number of dofs

        w_loops = wilson_loops(x, self.graph)
        w_lines = wilson_lines(x, self.graph)

        w_loops = nn.Conv(features=2, # for vertical and horizontal loops             
                          kernel_size=self.kernel_size, 
                          strides=1, 
                          padding='CIRCULAR')(w_loops) # PBC

        w_loops = nn.activation.leaky_relu(w_loops)

        # TODO reshape w_loops. w_lines is [vertical_edges, horizontal_edges]
        pointwise = w_lines * w_loops

        return x + pointwise

class InvariantCNN(nn.Module): 

    @nn.compact 
    def __call__(self, x): 
        """
        w_loops = wilson_loops(x)
        cnn1 = conv(w_loops) 
        cnn2 = conv(w_loops) 
        out1 = log(abs(elu(avg_pool(cnn1))))
        out2 = arg(softsign(avg_pool(cnn2)))/pi
        return out1, out2 (actually, 75% sure you just add them together)
        """

class GENN(nn.Module): 
    @nn.compact 
    def __call__(self, x): 
        # TODO 
        pass 

