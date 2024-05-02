"""
TODO 
- Equivariant layer 
- invariant layer 
- Gauge equivariant NN 
*** ALL WEIGHTS SHOULD BE REAL VALUED ***
"""

import flax.linen as nn 
import jax.numpy as np
import netket as nk
from src.wilson import * 
import math

class EquivariantCNN(nn.Module): 
    # include parameters like hidden size here: 
    kernel_size = (3, 3)

    @nn.compact 
    def __call__(self, loops, lines): 
        # Input should be of shape (B, N); B = batch size, N = number of dofs

        loops = np.expand_dims(loops, -1)
        w_loops = nn.Conv(features=2, # for vertical and horizontal loops             
                          kernel_size=self.kernel_size, 
                          strides=1, 
                          padding='CIRCULAR')(loops) # PBC

        w_loops = nn.activation.leaky_relu(w_loops)

        pointwise = lines * w_loops

        return pointwise

class InvariantCNN(nn.Module): 
    kernel_size = (3, 3)
    L: int = None

    @nn.compact 
    def __call__(self, x): 
        w_loops = nn.Conv(features=2, 
                          kernel_size=self.kernel_size, 
                          strides=1, 
                          padding='CIRCULAR')(x) 
        w_loops = nn.activation.leaky_relu(w_loops)
        cnn = nn.avg_pool(w_loops, window_shape=(self.L, self.L)) # out is (B, 1, 1, 2)
        cnn = cnn.reshape(-1, 2)
        cnn1 = np.log(cnn[0])
        cnn2 = np.pi * nn.soft_sign(cnn[1])
        # out1 = np.log(np.abs(nn.elu(nn.avg_pool(w_loops, window_shape=(self.L, self.L)))))
        # out2 = np.pi * nn.soft_sign(nn.avg_pool(w_loops, window_shape=(self.L, self.L))),
        return cnn1 + 1j * cnn2


class GENN(nn.Module): 

    graph: nk.graph.Graph = None

    def setup(self): 
        self.L = int(math.sqrt(self.graph.n_nodes))

    @nn.compact 
    def __call__(self, x): 
        w_loops, w_lines_left, w_lines_up = get_wilson_loops_and_lines(x, self.L)

        w_loops = w_loops.reshape(-1, self.L, self.L)
        w_lines_left = w_lines_left.reshape(-1, self.L, self.L)
        w_lines_up = w_lines_up.reshape(-1, self.L, self.L)

        lines = np.append(np.expand_dims(w_lines_left, -1), 
                          np.expand_dims(w_lines_up, -1),
                          axis=-1)

        eq = EquivariantCNN()(w_loops, lines)
        
        # x_ver = x[:, :self.L**2]
        # x_hor = x[:, self.L**2:]
        
        # x_ver = x_ver.reshape(-1, self.L, self.L)
        # x_hor = x_hor.reshape(-1, self.L, self.L)
        
        # x_reshaped = np.append(np.expand_dims(x_ver, -1), 
        #                        np.expand_dims(x_hor, -1),
        #                        axis=-1)

        # skip = x_reshaped + eq
        skip = x + eq.reshape(-1, 2 * self.L**2)
        w_loops, _, _  = get_wilson_loops_and_lines(skip, self.L)
        w_loops = np.expand_dims(w_loops.reshape(-1, self.L, self.L), -1)
        out = InvariantCNN(L=self.L)(w_loops)

        return out


