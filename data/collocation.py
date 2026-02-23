"""
data/collocation.py
===================
Generates collocation (training) points for the two-region MHD PINN problem.

The spatial domain is the unit interval y ∈ [-1, 1], split at y = 0:

  - Region 1 collocation:  y ∈ [0,  1]   (Newtonian fluid)
  - Region 2 collocation:  y ∈ [-1, 0]   (Casson fluid)

Special boundary/interface points:
  - x_right : y =  1  (right wall boundary condition)
  - x_left  : y = -1  (left  wall boundary condition)
  - x_iface : y =  0  (fluid-fluid interface condition)

All tensors have `requires_grad=True` so that PyTorch's automatic differentiation
(torch.autograd.grad) can compute the partial derivatives needed to evaluate the
PDE residuals inside the loss functions.

Usage
-----
    from data.collocation import generate_collocation_points
    data = generate_collocation_points(N=100)
    # data.x1      – Region 1 collocation points
    # data.x2      – Region 2 collocation points
    # data.x_iface – Interface point(s)
    # data.x_right – Right boundary point(s)
    # data.x_left  – Left boundary point(s)
"""

import numpy as np
import torch
from typing import NamedTuple

from config.params import DEVICE


class CollocationData(NamedTuple):
    """
    A lightweight container for all collocation and boundary tensors.

    Attributes
    ----------
    x1 : torch.Tensor, shape (N, 1)
        Collocation points for Region 1 (y ∈ [0, 1]).
    x2 : torch.Tensor, shape (N, 1)
        Collocation points for Region 2 (y ∈ [-1, 0]).
    x_right : torch.Tensor, shape (N, 1)
        Points at the right wall (y = 1), replicated to match x1 shape.
    x_left : torch.Tensor, shape (N, 1)
        Points at the left wall (y = -1), replicated to match x2 shape.
    x_iface : torch.Tensor, shape (N, 1)
        Points at the fluid-fluid interface (y = 0).
    """
    x1:      torch.Tensor
    x2:      torch.Tensor
    x_right: torch.Tensor
    x_left:  torch.Tensor
    x_iface: torch.Tensor


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a NumPy array to a float32 PyTorch tensor with gradient tracking.

    Parameters
    ----------
    arr    : np.ndarray of shape (N, 1)
    device : torch.device

    Returns
    -------
    torch.Tensor with requires_grad=True on the specified device.
    """
    return torch.tensor(arr, dtype=torch.float32, requires_grad=True).to(device)


def generate_collocation_points(N: int = 100, device: torch.device = DEVICE) -> CollocationData:
    """
    Create uniformly spaced collocation and boundary/interface tensors.

    The two fluid regions share N collocation points each.  The boundary and
    interface arrays are constant vectors of length N (all entries equal to the
    wall/interface coordinate), so that vectorised loss evaluation is consistent
    with the interior collocation arrays.

    Parameters
    ----------
    N      : int, optional (default 100)
        Number of collocation points in each region.
    device : torch.device, optional
        Target computation device (CPU or CUDA).

    Returns
    -------
    CollocationData
        Named tuple containing x1, x2, x_right, x_left, x_iface tensors.

    Notes
    -----
    Using np.linspace ensures a regular grid; random sampling (Latin Hypercube,
    Sobol, etc.) can be substituted here without changing the rest of the code.
    """
    # Region 1: y ∈ [0, 1]  – N uniformly spaced points
    x1_np = np.linspace(0, 1, N).reshape(-1, 1)

    # Region 2: y ∈ [-1, 0] – N uniformly spaced points
    x2_np = np.linspace(-1, 0, N).reshape(-1, 1)

    # Boundary at y = +1 (right wall), shape (N, 1)
    x_right_np = np.ones((N, 1), dtype=np.float32)

    # Boundary at y = -1 (left wall), shape (N, 1)
    x_left_np  = -np.ones((N, 1), dtype=np.float32)

    # Interface at y = 0, shape (N, 1)
    x_iface_np = np.zeros((N, 1), dtype=np.float32)

    return CollocationData(
        x1      = _to_tensor(x1_np,      device),
        x2      = _to_tensor(x2_np,      device),
        x_right = _to_tensor(x_right_np, device),
        x_left  = _to_tensor(x_left_np,  device),
        x_iface = _to_tensor(x_iface_np, device),
    )
