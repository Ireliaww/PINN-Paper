"""
pinn/network.py
===============
Neural network architectures for the PINN and GPINN solvers.

Architecture Overview
---------------------
Both networks share an identical topology: a fully-connected multi-layer
perceptron (MLP) with 4 hidden layers, each of width 20 neurons.

    Input  : y  ∈ ℝ¹   (spatial coordinate)
    Hidden : Linear(1→20) → Act → Linear(20→20) → Act → (×2 more)
    Output : ℝ¹⁰         (10 field variables)

The 10 output channels are arranged as:
    Index  0: u₁(y)   – axial velocity,      Region 1 (y ∈ [0, 1])
    Index  1: w₁(y)   – transverse velocity, Region 1
    Index  2: b₁(y)   – induced magnetic field component, Region 1
    Index  3: θ₁(y)   – temperature,         Region 1
    Index  4: g₁(y)   – species concentration, Region 1
    Index  5: u₂(y)   – axial velocity,      Region 2 (y ∈ [-1, 0])
    Index  6: w₂(y)   – transverse velocity, Region 2
    Index  7: b₂(y)   – induced magnetic field component, Region 2
    Index  8: θ₂(y)   – temperature,         Region 2
    Index  9: g₂(y)   – species concentration, Region 2

Both regions are solved simultaneously by a single network.  This design
choice simplifies the interface condition enforcement at y = 0.

Activation Function Choices
----------------------------
- **PINNNet** uses `tanh`:
    Smooth, bounded, with well-behaved gradients.  Standard choice for PINNs
    since `tanh` approximates the sigmoidal shape expected in boundary-layer
    solutions, and its higher-order derivatives are non-zero (required because
    the PDE involves up to 2nd-order derivatives).

- **GPINNNet** uses `softmax`:
    Following the GPINN literature, a non-standard activation is tested here.
    Note that softmax is applied channel-wise; the key difference from tanh
    is that it enforces a probability-simplex normalisation across neurons,
    which can help with gradient-enhanced training.

References
----------
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed
neural networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations.
Journal of Computational Physics, 378, 686-707.

Yu, J. et al. (2022). Gradient-enhanced physics-informed neural networks for
forward and inverse PDE problems. Computer Methods in Applied Mechanics and
Engineering, 393, 114823.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    """
    Shared MLP backbone used by both PINNNet and GPINNNet.

    Parameters
    ----------
    activation : callable
        Activation function applied after each hidden layer.
        Must accept a torch.Tensor and return a torch.Tensor of the same shape.

    Architecture
    ------------
    Linear(1, 20) → act → Linear(20, 20) → act → Linear(20, 20) → act
    → Linear(20, 20) → act → Linear(20, 10)
    """

    def __init__(self, activation):
        super(BaseNet, self).__init__()

        # Hidden layers: 4 layers of width 20
        self.fc1 = nn.Linear(1,  20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)

        # Output layer: maps to the 10 field variables (5 per region)
        self.fc_out = nn.Linear(20, 10)

        # Store activation for use in forward pass
        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor, shape (N, 1)
            Spatial coordinate y, with requires_grad=True for auto-diff.

        Returns
        -------
        torch.Tensor, shape (N, 10)
            Predicted values for all 10 field variables at the given points.
        """
        h = self._activation(self.fc1(x))
        h = self._activation(self.fc2(h))
        h = self._activation(self.fc3(h))
        h = self._activation(self.fc4(h))
        return self.fc_out(h)

    def __repr__(self) -> str:
        act_name = getattr(self._activation, "__name__",
                   getattr(self._activation, "__class__", self._activation).__name__)
        return (f"{self.__class__.__name__}("
                f"layers=[1,20,20,20,20,10], "
                f"activation={act_name})")


class PINNNet(BaseNet):
    """
    Standard Physics-Informed Neural Network (PINN) with tanh activation.

    The PINN loss consists only of the PDE residuals and boundary/interface
    conditions (no gradient-enhancement).

    Activation: tanh
        - Analytic for all orders (no discontinuous gradients).
        - Non-vanishing higher derivatives (critical for 2nd-order PDEs).
        - Maps ℝ → (-1, 1), naturally conditioning the hidden features.
    """

    def __init__(self):
        super(PINNNet, self).__init__(activation=torch.tanh)


class GPINNNet(BaseNet):
    """
    Gradient-enhanced Physics-Informed Neural Network (GPINN) with softmax activation.

    In addition to the standard PDE residuals, GPINN also penalises the
    *spatial derivative* of each PDE residual (∂R/∂y = 0), which provides
    additional gradient information and can accelerate convergence.

    Activation: softmax (applied along the feature dimension)
        Normalises the activations across the 20 neurons, which can improve
        the training behaviour for gradient-enhanced losses.
    """

    def __init__(self):
        # softmax along dim=-1 (across the 20 hidden neurons)
        super(GPINNNet, self).__init__(
            activation=lambda x: F.softmax(x, dim=-1)
        )
