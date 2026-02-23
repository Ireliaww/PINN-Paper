"""
config/params.py
================
Centralised configuration for the two-region MHD channel-flow PINN/GPINN study.

Physical Problem
----------------
We consider a fully-developed, laminar, electrically-conducting fluid flow between
two infinite parallel plates located at y = -1 (left wall) and y = +1 (right wall).
The channel is divided at the interface y = 0 into two distinct fluid regions:

  - Region 1  (y ∈ [0, 1])  : Newtonian fluid.
  - Region 2  (y ∈ [-1, 0]) : Casson (non-Newtonian) fluid with Casson parameter β.

An external magnetic field is applied at an inclination angle θ to the y-axis.
The system is also subject to rotation (Coriolis force, parameter ω), buoyancy
(Grashof number Gr), thermal radiation (Rd), and chemical reaction (η, Sc).

For each field variable (u, w, b, θ, g), indices 1 and 2 refer to Region 1 and
Region 2 respectively.

Governing PDEs (non-dimensionalised)
--------------------------------------
The governing coupled ODEs in each region and the corresponding boundary/interface
conditions are enforced as soft constraints via the PINN/GPINN loss function.
See pinn/losses.py for the explicit PDE residual forms.

References
----------
This problem setup follows standard coupled MHD channel-flow literature.
Refer to README.md for the full citation list.
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Compute Device
# ---------------------------------------------------------------------------
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Physical Parameters – Region 1 (Newtonian fluid, y ∈ [0, 1])
# ---------------------------------------------------------------------------

Ha1    = 1      # Hartmann number (Region 1): ratio of electromagnetic body force
                # to viscous force.  Ha² = σ·B₀²·L² / μ
Gr1    = 5      # Grashof number (Region 1): ratio of buoyancy to viscous force.
                # Drives natural convection.
Rd1    = 1      # Radiation parameter (Region 1): relative contribution of thermal
                # radiation heat flux to conductive heat flux.
kb1    = 0.1    # Nonlinear thermal buoyancy coefficient (Region 1).
G1     = 1      # Dimensionless pressure gradient applied to Region 1.
omega1 = 0.5    # Rotation parameter (Region 1): Taylor-number based Coriolis term.
Pr1    = 7.43   # Prandtl number (Region 1): ratio of momentum diffusivity to
                # thermal diffusivity.  Pr = μ·cₚ / k
Ec1    = 0.0017 # Eckert number (Region 1): ratio of kinetic energy to enthalpy
                # difference.  Governs viscous dissipation.
Sc1    = 1      # Schmidt number (Region 1): ratio of momentum diffusivity to
                # mass diffusivity.
eta1   = 0.8    # Chemical reaction rate parameter (Region 1).
Rm1    = 0.7    # Magnetic Reynolds number (Region 1): ratio of magnetic advection
                # to magnetic diffusion.

# ---------------------------------------------------------------------------
# Physical Parameters – Region 2 (Casson fluid, y ∈ [-1, 0])
# ---------------------------------------------------------------------------

Ha2    = 1      # Hartmann number (Region 2).
Gr2    = 5      # Grashof number (Region 2).
Rd2    = 1      # Radiation parameter (Region 2).
kb2    = 0.1    # Nonlinear thermal buoyancy coefficient (Region 2).
G2     = 5      # Dimensionless pressure gradient applied to Region 2.
omega2 = 0.5    # Rotation parameter (Region 2).
Pr2    = 10     # Prandtl number (Region 2).
Ec2    = 0.005  # Eckert number (Region 2).
Sc2    = 1      # Schmidt number (Region 2).
eta2   = 0.8    # Chemical reaction rate parameter (Region 2).
Rm2    = 0.7    # Magnetic Reynolds number (Region 2).

# ---------------------------------------------------------------------------
# Shared Physical Parameters
# ---------------------------------------------------------------------------

beta   = 0.5    # Casson fluid parameter (β): characterises the non-Newtonian
                # yield stress.  The (1 + 1/β) factor appears in the momentum
                # equations of Region 2.  As β → ∞, the fluid becomes Newtonian.
beta1  = beta   # Alias used in interface conditions.
K      = 2      # Permeability parameter of the porous medium.
theta  = 60 * np.pi / 180  # Inclination angle of the applied magnetic field to
                            # the y-axis (radians).  θ = 60° here.
n      = 2      # Chemical reaction order (homogeneous reaction exponent).

# Interface coupling parameters (at y = 0)
alpha   = 0.6   # Viscosity ratio parameter at the interface.
delta   = 0.5   # Electrical conductivity ratio at the interface.
phi     = 0.2   # Mass diffusivity ratio at the interface.
epsilon = 0.2   # Thermal conductivity ratio at the interface.
gamma   = 0.5   # Ratio related to magnetic diffusivity at the interface.

# ---------------------------------------------------------------------------
# Sweep Variables
# ---------------------------------------------------------------------------
# Lambda (λ = cos θ) can be fixed or swept.  The original code used a list
# [0.5] enabling easy extension to multiple values.
LAMDAS = [0.5]  # List of λ = cos(θ) values to sweep over.

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------

N_COLLOC      = 100      # Number of collocation points per region (uniform grid).
ADAM_EPOCHS   = 200      # Number of Adam optimiser epochs in Phase 1.
LBFGS_MAX_ITER = 1000    # Maximum L-BFGS iterations in Phase 2.
LR_ADAM       = 1e-3     # Adam learning rate.

# Loss thresholds at which training epoch and wall-clock time are recorded.
# Useful for convergence comparison between PINN and GPINN.
LOSS_THRESHOLDS = [1e-4, 1e-6, 1e-8]

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------

MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
