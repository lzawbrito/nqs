"""
TODO 
- Equivariant layer 
- invariant layer 
- Gauge equivariant NN 
*** ALL WEIGHTS SHOULD BE REAL VALUED ***
"""

import flax.linen as nn 
import jax.numpy as np
import jax
import netket as nk
from src.wilson import * 
import math

from typing import Union, Any

from flax import linen as nn
from jax.nn.initializers import normal

from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket.utils.group import PermutationGroup
from netket import nn as nknn

default_kernel_init = normal(stddev=0.01)


class EquivariantCNN(nn.Module): 
    # include parameters like hidden size here: 
    kernel_size = 3

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
    kernel_size = 3
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

        cnn1 = nn.elu(cnn[:, 0])
        cnn2 = np.pi * nn.soft_sign(cnn[:, 1])

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

        skip = x + eq.reshape(-1, 2 * self.L**2)
        # print(skip.shape)
        w_loops, _, _  = get_wilson_loops_and_lines(skip, self.L)
        # print(w_loops.shape)
        w_loops = np.expand_dims(w_loops.reshape(-1, self.L, self.L), -1)
        out = InvariantCNN(L=self.L)(w_loops)
        # jax.debug.print("out {bar}", bar=out)
        return out

class RBM(nn.Module):
    r"""A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between.
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = np.sum(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                self.param_dtype,
            )
            out_bias = np.dot(input, v_bias)
            return x + out_bias
        else:
            return x


class GERBIL(nn.Module): 

    graph: nk.graph.Graph = None

    def setup(self): 
        self.L = int(math.sqrt(self.graph.n_nodes))

    @nn.compact 
    def __call__(self, x): 

        sig = x[:, 2 * (self.L**2):]  # Extract the first slice
        # sig = sig.reshape(-1, self.L, self.L)
        # sig = np.expand_dims(sig, -1)
        sig = sig.reshape(-1, self.L, self.L, 1)
        x = x[:, :2 * (self.L**2)]

        w_loops, w_lines_left, w_lines_up = get_wilson_loops_and_lines(x, self.L)
        w_loops = w_loops.reshape(-1, self.L, self.L)

        w_lines_left = w_lines_left.reshape(-1, self.L, self.L)
        w_lines_up = w_lines_up.reshape(-1, self.L, self.L)

        lines = np.append(np.expand_dims(w_lines_left, -1), 
                          np.expand_dims(w_lines_up, -1),
                          axis=-1)

        eq = EquivariantCNN()(w_loops, lines)

        skip1 = x + eq.reshape(-1, 2 * self.L**2)
        skip2 = skip1 * sig
        # print(skip.shape)
        w_loops, _, _  = get_wilson_loops_and_lines(skip2, self.L)
        # print(w_loops.shape)
        w_loops = np.expand_dims(w_loops.reshape(-1, self.L, self.L), -1)
        out = InvariantCNN(L=self.L)(w_loops)

        # RBM Layer on sigs
        rbm_layer = nn.Dense(
            name="RBM",
            features=int((max(1, self.L/10)) * sig.shape[-1]),
            param_dtype=np.float64,
            precision=None,
            use_bias=True,
            kernel_init=default_kernel_init,
            bias_init=default_kernel_init,
        )(sig)
        rbm_output = nknn.log_cosh(rbm_layer)
        rbm_output = np.sum(rbm_output, axis=(-2, -3))

        v_bias = self.param(
            "visible_bias",
            default_kernel_init,
            (rbm_output.shape[-1],),
            np.float64,
        )

        rbm_output = rbm_output + np.dot(rbm_output, v_bias)

        #rbm_output = np.log(np.cosh(rbm_output))
        out = out + rbm_output
        # jax.debug.print("out {bar}", bar=out)
        return out


