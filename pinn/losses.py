"""
pinn/losses.py
==============
PDE residual and boundary/interface loss functions for the two-region MHD PINN.

Physical Background
-------------------
The governing system consists of five coupled second-order ODEs in each region,
describing: axial velocity (u), transverse velocity (w), induced magnetic field (b),
temperature (θ), and species concentration (g).

All equations are expressed in non-dimensional form.  The key dimensionless
groups and their physical roles are defined in config/params.py.

Equation Structure
------------------
For **Region 1** (Newtonian fluid, y ∈ [0, 1]), letting λ = cos(θ):

  (M1) u₁'' + G₁ - Ha₁²(K + λu₁)λ - 2ω₁w₁ + Gr₁(θ₁ + kb₁θ₁²) = 0
       └ viscous ┘  └ pressure ┘  └─ Lorentz drag ─┘  └ Coriolis ┘  └── buoyancy ──┘

  (M2) w₁'' + G₁ - Ha₁²(λ² + (b₁ + √(1-λ²))²) + 2ω₁u₁ = 0
       └ viscous ┘  └ pressure ┘  └────── Lorentz drag ──────────┘  └ Coriolis ┘

  (B1) b₁'' + λ Rm₁ u₁' = 0
       └─ induction ─┘  └── advection ──┘

  (E1) (1 + 4Rd₁) θ₁'' + Pr₁ Ec₁ (u₁'² + w₁'²) + Ha₁² Pr₁ Ec₁ (K + λu₁)² = 0
       └────── conduction + radiation ────┘  └── viscous dissipation ───┘  └── Joule heating ──┘

  (C1) g₁'' - Sc₁ η₁ g₁ⁿ = 0
       └─ diffusion ─┘  └── chemical reaction ──┘

For **Region 2** (Casson fluid, y ∈ [-1, 0]), the momentum equations gain the
Casson factor (1 + 1/β):

  (M1') (1+1/β) u₂'' + G₂ - Ha₂²(K + λu₂)λ - 2ω₂w₂ + Gr₂(θ₂ + kb₂θ₂²) = 0
  (M2') (1+1/β) w₂'' + G₂ - Ha₂²(λ² + (b₂ + √(1-λ²))²) + 2ω₂u₂ = 0
  (B2)  b₂'' + λ Rm₂ u₂' = 0
  (E2)  (1 + 4Rd₂) θ₂'' + Pr₂ Ec₂ (u₂'² + w₂'²) + Ha₂² Pr₂ Ec₂ (K + λu₂)² = 0
  (C2)  g₂'' - Sc₂ η₂ g₂ⁿ = 0

Boundary Conditions
-------------------
At y = +1 (right wall, xright):
    u₁(1) = 1, w₁(1) = 0, b₁(1) = 0, θ₁(1) = 1, g₁(1) = 1

At y = -1 (left wall, xleft):
    u₂(-1) = 0, w₂(-1) = 0, b₂(-1) = 0, θ₂(-1) = 0, g₂(-1) = 0

At y = 0 (interface, xface):
    u₁ = u₂,   w₁ = w₂,   b₁ = b₂,   θ₁ = θ₂,   g₁ = g₂         (continuity)
    u₁' = (1+1/β) u₂',   w₁' = (1+1/β) w₂'                        (shear stress)
    b₁' = δ β₁ γ b₂'                                                 (magnetic flux)
    θ₁' = (β₁/ε) θ₂'                                                (heat flux)
    g₁' = (β₁/φ) g₂'                                                (mass flux)

GPINN Enhancement
-----------------
GPINN additionally enforces the *y-derivative* of every PDE residual term to
be zero, i.e. ∂R/∂y = 0 for each residual R.  This doubles the number of
PDE constraints but provides extra gradient information that accelerates training.

Loss Function
-------------
All residuals are enforced via MSE loss (Mean Squared Error = L₂ penalty):
    L = MSE(R, 0)  for each residual R.
This is the standard 'soft constraint' approach in PINN methods.
"""

import torch
import torch.nn as nn
from config import params as P

# Mean Squared Error loss used for all soft constraints
mse_loss = nn.MSELoss()


# ---------------------------------------------------------------------------
# Utility: Automatic Differentiation Helpers
# ---------------------------------------------------------------------------

def _grad(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the first-order derivative of `output` with respect to `x`
    using PyTorch automatic differentiation.

    This is a thin wrapper around torch.autograd.grad that keeps the
    computational graph for higher-order derivative computations.

    Parameters
    ----------
    output : torch.Tensor
        Scalar field evaluated at x (shape (N, 1)).
    x : torch.Tensor
        Input spatial coordinate (shape (N, 1)), must have requires_grad=True.

    Returns
    -------
    torch.Tensor, shape (N, 1)
        d(output)/dx at every collocation point.
    """
    return torch.autograd.grad(
        output.sum(), x,
        create_graph=True,   # retain graph for higher-order diffs
        retain_graph=True,
    )[0]


def _deriv(field: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Compute an arbitrary-order derivative of `field` w.r.t. `x`.

    Parameters
    ----------
    field : torch.Tensor, shape (N, 1)
        The scalar field to differentiate.
    x     : torch.Tensor, shape (N, 1)
        The independent variable.  Must have requires_grad=True.
    order : int (1, 2, or 3)
        Desired derivative order.

    Returns
    -------
    torch.Tensor, shape (N, 1)
        The `order`-th derivative of `field` w.r.t. `x`.
    """
    d = field
    for _ in range(order):
        d = _grad(d, x)
    return d


# ---------------------------------------------------------------------------
# PDE Residuals – Region 1 (Newtonian fluid)
# ---------------------------------------------------------------------------

def pde_residuals_region1(net: nn.Module, x: torch.Tensor, lamda: float) -> list:
    """
    Compute the five PDE residuals for Region 1 (Newtonian, y ∈ [0, 1]).

    The residuals correspond to equations (M1), (M2), (B1), (E1), (C1)
    described in the module docstring.  Each residual should equal zero
    at every collocation point if the network perfectly satisfies the PDE.

    Parameters
    ----------
    net   : nn.Module  (PINNNet or GPINNNet)
        Neural network providing the field predictions.
    x     : torch.Tensor, shape (N, 1)
        Collocation points in Region 1 (y ∈ [0, 1]).
    lamda : float
        λ = cos(θ), the cosine of the magnetic field inclination angle.

    Returns
    -------
    list of 5 torch.Tensor, each shape (N, 1)
        [R_u1, R_w1, R_b1, R_theta1, R_g1]
    """
    out = net(x)
    u1, w1, b1, theta1, g1 = [out[:, i].reshape(-1, 1) for i in range(5)]

    # First and second derivatives via automatic differentiation
    du   = _deriv(u1, x, 1);  d2u = _deriv(u1, x, 2)
    dw   = _deriv(w1, x, 1);  d2w = _deriv(w1, x, 2)
    db   = _deriv(b1, x, 1);  d2b = _deriv(b1, x, 2)
    dtheta = _deriv(theta1, x, 1); d2theta = _deriv(theta1, x, 2)
    dg   = _deriv(g1,  x, 1);  d2g = _deriv(g1,  x, 2)

    # (M1) Momentum in x-direction (axial velocity u₁)
    # u₁'' + G₁ - Ha₁²(K + λu₁)λ - 2ω₁w₁ + Gr₁(θ₁ + kb₁θ₁²) = 0
    R_u = d2u + P.G1 - P.Ha1**2 * (P.K + u1 * lamda) * lamda \
          - 2 * P.omega1 * w1 + P.Gr1 * (theta1 + P.kb1 * theta1**2)

    # (M2) Momentum in z-direction (transverse velocity w₁)
    # w₁'' + G₁ - Ha₁²(λ² + (b₁ + √(1-λ²))²) + 2ω₁u₁ = 0
    # The term √(1-λ²) = sin(θ) represents the y-component of the applied field.
    R_w = d2w + P.G1 - P.Ha1**2 * (lamda**2 + (b1 + torch.sqrt(x*0+1 - lamda**2))**2) \
          + 2 * P.omega1 * u1

    # (B1) Induction equation for induced magnetic field b₁
    # b₁'' + λ Rm₁ u₁' = 0
    R_b = d2b + lamda * P.Rm1 * du

    # (E1) Energy equation (temperature θ₁) with radiation and Joule/viscous heating
    # (1 + 4Rd₁) θ₁'' + Pr₁ Ec₁ (u₁'² + w₁'²) + Ha₁² Pr₁ Ec₁ (K + λu₁)² = 0
    R_theta = (1 + 4*P.Rd1) * d2theta \
              + P.Pr1 * P.Ec1 * (du**2 + dw**2) \
              + P.Ha1**2 * P.Pr1 * P.Ec1 * (P.K + u1 * lamda)**2

    # (C1) Species diffusion with nth-order homogeneous chemical reaction
    # g₁'' - Sc₁ η₁ g₁ⁿ = 0
    R_g = d2g - P.Sc1 * P.eta1 * g1**P.n

    return [R_u, R_w, R_b, R_theta, R_g]


# ---------------------------------------------------------------------------
# PDE Residuals – Region 2 (Casson fluid)
# ---------------------------------------------------------------------------

def pde_residuals_region2(net: nn.Module, x: torch.Tensor, lamda: float) -> list:
    """
    Compute the five PDE residuals for Region 2 (Casson, y ∈ [-1, 0]).

    The Casson factor (1 + 1/β) modifies the momentum equations, increasing
    effective viscosity and thus stiffening the velocity profiles compared to
    the Newtonian region.  The energy and concentration equations retain the
    same structure as Region 1 (with Region-2 parameters).

    Parameters
    ----------
    net   : nn.Module  (PINNNet or GPINNNet)
    x     : torch.Tensor, shape (N, 1)  – collocation points in y ∈ [-1, 0]
    lamda : float  – λ = cos(θ)

    Returns
    -------
    list of 5 torch.Tensor, each shape (N, 1)
        [R_u2, R_w2, R_b2, R_theta2, R_g2]
    """
    out = net(x)
    u2, w2, b2, theta2, g2 = [out[:, i+5].reshape(-1, 1) for i in range(5)]

    # Derivatives
    du   = _deriv(u2, x, 1);  d2u = _deriv(u2, x, 2)
    dw   = _deriv(w2, x, 1);  d2w = _deriv(w2, x, 2)
    db   = _deriv(b2, x, 1);  d2b = _deriv(b2, x, 2)
    dtheta = _deriv(theta2, x, 1); d2theta = _deriv(theta2, x, 2)
    dg   = _deriv(g2,  x, 1);  d2g = _deriv(g2,  x, 2)

    # Casson effective viscosity factor
    casson = 1.0 + 1.0 / P.beta

    # (M1') Axial momentum with Casson correction
    # (1+1/β) u₂'' + G₂ - Ha₂²(K + λu₂)λ - 2ω₂w₂ + Gr₂(θ₂ + kb₂θ₂²) = 0
    R_u = casson * d2u + P.G2 \
          - P.Ha2**2 * (P.K + u2 * lamda) * lamda \
          - 2 * P.omega2 * w2 + P.Gr2 * (theta2 + P.kb2 * theta2**2)

    # (M2') Transverse momentum with Casson correction
    # (1+1/β) w₂'' + G₂ - Ha₂²(λ² + (b₂ + √(1-λ²))²) + 2ω₂u₂ = 0
    R_w = casson * d2w + P.G2 \
          - P.Ha2**2 * (lamda**2 + (b2 + torch.sqrt(x*0+1 - lamda**2))**2) \
          + 2 * P.omega2 * u2

    # (B2) Induction equation (same form as Region 1)
    R_b = d2b + lamda * P.Rm2 * du

    # (E2) Energy equation for Region 2
    R_theta = (1 + 4*P.Rd2) * d2theta \
              + P.Pr2 * P.Ec2 * (du**2 + dw**2) \
              + P.Ha2**2 * P.Pr2 * P.Ec2 * (P.K + u2 * lamda)**2

    # (C2) Concentration equation for Region 2
    R_g = d2g - P.Sc2 * P.eta2 * g2**P.n

    return [R_u, R_w, R_b, R_theta, R_g]


# ---------------------------------------------------------------------------
# GPINN Gradient-Enhanced Residuals (∂R/∂y = 0)
# ---------------------------------------------------------------------------

def gpde_residuals_region1(net: nn.Module, x: torch.Tensor, lamda: float) -> list:
    """
    Compute the y-derivatives of the Region 1 PDE residuals (GPINN enhancement).

    GPINN adds ∂R_i/∂y = 0 as additional loss terms.  This is equivalent to
    requiring that the solution satisfies the PDE not just pointwise but also
    in a weighted Galerkin sense over the derivative space.

    Parameters
    ----------
    net, x, lamda : see pde_residuals_region1

    Returns
    -------
    list of 5 torch.Tensor  –  [∂R_u1/∂y, ∂R_w1/∂y, ..., ∂R_g1/∂y]
    """
    out = net(x)
    u1, w1, b1, theta1, g1 = [out[:, i].reshape(-1, 1) for i in range(5)]

    du   = _deriv(u1, x, 1); d2u = _deriv(u1, x, 2); d3u = _deriv(u1, x, 3)
    dw   = _deriv(w1, x, 1); d2w = _deriv(w1, x, 2); d3w = _deriv(w1, x, 3)
    db   = _deriv(b1, x, 1); d2b = _deriv(b1, x, 2); d3b = _deriv(b1, x, 3)
    dtheta = _deriv(theta1, x, 1)
    d2theta = _deriv(theta1, x, 2); d3theta = _deriv(theta1, x, 3)
    dg   = _deriv(g1,  x, 1); d2g = _deriv(g1,  x, 2); d3g = _deriv(g1,  x, 3)

    # ∂/∂y of (M1): u''' - Ha₁²λ²u' - 2ω₁w' + Gr₁(θ' + 2kb₁θ₁θ')
    gR_u = d3u - P.Ha1**2 * lamda**2 * du \
           - 2 * P.omega1 * dw \
           + P.Gr1 * (dtheta + 2 * P.kb1 * theta1 * dtheta)

    # ∂/∂y of (M2): w''' - 2Ha₁²(b₁b₁' + √(1-λ²)b₁') + 2ω₁u'
    gR_w = d3w \
           - 2 * P.Ha1**2 * (b1 * db + torch.sqrt(x*0+1 - lamda**2) * db) \
           + 2 * P.omega1 * du

    # ∂/∂y of (B1): b''' + λ Rm₁ u''
    gR_b = d3b + lamda * P.Rm1 * d2u

    # ∂/∂y of (E1) with chain rule on viscous dissipation and Joule terms
    gR_theta = (1 + 4*P.Rd1) * d3theta \
               + P.Pr1 * P.Ec1 * (2*du*d2u + 2*dw*d2w) \
               + 2 * P.Ha1**2 * P.Pr1 * P.Ec1 * (P.K * lamda * du + lamda**2 * u1 * du)

    # ∂/∂y of (C1): g''' - Sc₁ η₁ n g^(n-1) g'
    gR_g = d3g - P.Sc1 * P.eta1 * P.n * g1**(P.n - 1) * dg

    return [gR_u, gR_w, gR_b, gR_theta, gR_g]


def gpde_residuals_region2(net: nn.Module, x: torch.Tensor, lamda: float) -> list:
    """
    Compute the y-derivatives of the Region 2 PDE residuals (GPINN enhancement).

    Parameters
    ----------
    net, x, lamda : see pde_residuals_region2

    Returns
    -------
    list of 5 torch.Tensor  –  [∂R_u2/∂y, ∂R_w2/∂y, ..., ∂R_g2/∂y]
    """
    out = net(x)
    u2, w2, b2, theta2, g2 = [out[:, i+5].reshape(-1, 1) for i in range(5)]

    du   = _deriv(u2, x, 1); d2u = _deriv(u2, x, 2); d3u = _deriv(u2, x, 3)
    dw   = _deriv(w2, x, 1); d2w = _deriv(w2, x, 2); d3w = _deriv(w2, x, 3)
    db   = _deriv(b2, x, 1); d2b = _deriv(b2, x, 2); d3b = _deriv(b2, x, 3)
    dtheta = _deriv(theta2, x, 1)
    d2theta = _deriv(theta2, x, 2); d3theta = _deriv(theta2, x, 3)
    dg   = _deriv(g2,  x, 1); d2g = _deriv(g2,  x, 2); d3g = _deriv(g2,  x, 3)

    casson = 1.0 + 1.0 / P.beta

    # ∂/∂y of (M1'): (1+1/β)u''' - Ha₂²λ²u' - 2ω₂w' + Gr₂(θ' + 2kb₂θ₂θ')
    gR_u = casson * d3u \
           - P.Ha2**2 * lamda**2 * du \
           - 2 * P.omega2 * dw \
           + P.Gr2 * (dtheta + 2 * P.kb2 * theta2 * dtheta)

    # ∂/∂y of (M2'): (1+1/β)w''' - 2Ha₂²(b₂ + √(1-λ²))b₂' + 2ω₂u'
    gR_w = casson * d3w \
           - 2 * P.Ha2**2 * (b2 + torch.sqrt(x*0+1 - lamda**2)) * db \
           + 2 * P.omega2 * du

    # ∂/∂y of (B2)
    gR_b = d3b + lamda * P.Rm2 * d2u

    # ∂/∂y of (E2)
    gR_theta = (1 + 4*P.Rd2) * d3theta \
               + P.Pr2 * P.Ec2 * (2*du*d2u + 2*dw*d2w) \
               + 2 * P.Ha2**2 * P.Pr2 * P.Ec2 * du * (P.K * lamda + lamda**2 * u2)

    # ∂/∂y of (C2)
    gR_g = d3g - P.Sc2 * P.eta2 * P.n * g2**(P.n - 1) * dg

    return [gR_u, gR_w, gR_b, gR_theta, gR_g]


# ---------------------------------------------------------------------------
# Boundary Condition Losses
# ---------------------------------------------------------------------------

def bc_loss_right(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Boundary condition loss at the right wall, y = +1.

    Enforces: u₁ = 1, w₁ = 0, b₁ = 0, θ₁ = 1, g₁ = 1.

    These Dirichlet conditions correspond to the physical state of the right plate:
    unit velocity (moving wall), no transverse velocity, no induced field, unit
    temperature, and unit species concentration.

    Parameters
    ----------
    net : nn.Module
    x   : torch.Tensor of shape (N_bc, 1) containing y = 1 uniformly.

    Returns
    -------
    torch.Tensor  –  scalar MSE loss.
    """
    out = net(x)
    u1, w1, b1, theta1, g1 = [out[:, i].reshape(-1, 1) for i in range(5)]

    ones  = torch.ones_like(u1)
    zeros = torch.zeros_like(u1)

    return (mse_loss(u1, ones)   +
            mse_loss(w1, zeros)  +
            mse_loss(b1, zeros)  +
            mse_loss(theta1, ones) +
            mse_loss(g1, ones))


def bc_loss_left(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Boundary condition loss at the left wall, y = -1.

    Enforces: u₂ = 0, w₂ = 0, b₂ = 0, θ₂ = 0, g₂ = 0.

    The left plate is stationary (no-slip), insulating, isothermal (cold), and
    impermeable to species.

    Parameters
    ----------
    net : nn.Module
    x   : torch.Tensor of shape (N_bc, 1) containing y = -1 uniformly.

    Returns
    -------
    torch.Tensor  –  scalar MSE loss.
    """
    out = net(x)
    u2, w2, b2, theta2, g2 = [out[:, i+5].reshape(-1, 1) for i in range(5)]
    zeros = torch.zeros_like(u2)

    return (mse_loss(u2, zeros) + mse_loss(w2, zeros) + mse_loss(b2, zeros) +
            mse_loss(theta2, zeros) + mse_loss(g2, zeros))


def interface_loss(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Interface condition loss at y = 0 (fluid-fluid interface).

    Enforces both value continuity and flux continuity between the two regions.

    Value continuity (10 conditions → 5 unique pairs):
        u₁(0) = u₂(0),  w₁(0) = w₂(0),  b₁(0) = b₂(0),  θ₁(0) = θ₂(0),  g₁(0) = g₂(0)

    Flux continuity (gradient jump conditions):
        u₁'(0) = (1 + 1/β) u₂'(0)    [shear stress balance, Newtonian vs Casson]
        w₁'(0) = (1 + 1/β) w₂'(0)    [shear stress balance]
        b₁'(0) = δ·β₁·γ · b₂'(0)     [magnetic field normal flux jump]
        θ₁'(0) = (β₁/ε) · θ₂'(0)     [heat flux continuity with conductivity ratio]
        g₁'(0) = (β₁/φ) · g₂'(0)     [mass flux continuity with diffusivity ratio]

    Parameters
    ----------
    net : nn.Module
    x   : torch.Tensor of shape (N_ic, 1) containing y = 0 uniformly.

    Returns
    -------
    torch.Tensor  –  scalar MSE loss.
    """
    out = net(x)
    u1, w1, b1, theta1, g1 = [out[:, i].reshape(-1, 1) for i in range(5)]
    u2, w2, b2, theta2, g2 = [out[:, i+5].reshape(-1, 1) for i in range(5)]

    # First derivatives
    du1 = _grad(u1, x);   du2 = _grad(u2, x)
    dw1 = _grad(w1, x);   dw2 = _grad(w2, x)
    db1 = _grad(b1, x);   db2 = _grad(b2, x)
    dth1 = _grad(theta1, x); dth2 = _grad(theta2, x)
    dg1  = _grad(g1,  x); dg2  = _grad(g2,  x)

    casson = 1.0 + 1.0 / P.beta

    zeros = torch.zeros_like(u1)

    # Flux jump conditions (residual form: LHS - RHS = 0)
    jmp_u  = du1  - casson * du2
    jmp_w  = dw1  - casson * dw2
    jmp_b  = db1  - P.delta * P.beta1 * P.gamma * db2
    jmp_th = dth1 - (P.beta1 / P.epsilon) * dth2
    jmp_g  = dg1  - (P.beta1 / P.phi)     * dg2

    # Value continuity residuals
    val_u  = u1  - u2
    val_w  = w1  - w2
    val_b  = b1  - b2
    val_th = theta1 - theta2
    val_g  = g1  - g2

    return (mse_loss(jmp_u,  zeros) + mse_loss(jmp_w,  zeros) +
            mse_loss(jmp_b,  zeros) + mse_loss(jmp_th, zeros) +
            mse_loss(jmp_g,  zeros) +
            mse_loss(val_u,  zeros) + mse_loss(val_w,  zeros) +
            mse_loss(val_b,  zeros) + mse_loss(val_th, zeros) +
            mse_loss(val_g,  zeros))


# ---------------------------------------------------------------------------
# Aggregate Loss Functions
# ---------------------------------------------------------------------------

def _residual_loss(residuals: list) -> torch.Tensor:
    """Sum MSE losses for a list of residual tensors."""
    zeros_list = [torch.zeros_like(r) for r in residuals]
    return sum(mse_loss(r, z) for r, z in zip(residuals, zeros_list))


def total_loss_pinn(net: nn.Module, data, lamda: float) -> torch.Tensor:
    """
    Compute the total PINN loss for one training step.

    Total Loss = L_pde1 + L_pde2 + L_bc_right + L_bc_left + L_interface

    where:
        L_pde1 = sum of MSE(R_i, 0) for Region 1 residuals (weighted 0.1)
        L_pde2 = sum of MSE(R_i, 0) for Region 2 residuals (weighted 0.1)
        L_bc_* = boundary condition MSE losses
        L_interface = interface condition MSE loss

    Note: The 0.1 weight on interior PDE terms down-weights the PDE loss
    relative to the boundary conditions in early training, improving stability.

    Parameters
    ----------
    net   : PINNNet
    data  : CollocationData namedtuple  (see data/collocation.py)
    lamda : float  –  λ = cos(θ)

    Returns
    -------
    torch.Tensor  –  scalar total loss.
    """
    pde1 = _residual_loss(pde_residuals_region1(net, data.x1, lamda))
    pde2 = _residual_loss(pde_residuals_region2(net, data.x2, lamda))
    bc_r = bc_loss_right(net, data.x_right)
    bc_l = bc_loss_left (net, data.x_left)
    ic   = interface_loss(net, data.x_iface)

    return 0.1 * pde1 + 0.1 * pde2 + bc_r + bc_l + ic


def total_loss_gpinn(net: nn.Module, data, lamda: float) -> torch.Tensor:
    """
    Compute the total GPINN loss for one training step.

    Total Loss = L_pde1 + L_gpde1 + L_pde2 + L_gpde2
               + L_bc_right + L_bc_left + L_interface

    The gradient-enhanced terms L_gpde1 and L_gpde2 provide additional
    first-derivative constraints that sharpen the PDE satisfaction and
    typically lead to faster convergence to low loss values.

    Parameters
    ----------
    net   : GPINNNet
    data  : CollocationData namedtuple
    lamda : float

    Returns
    -------
    torch.Tensor  –  scalar total loss.
    """
    pde1  = _residual_loss(pde_residuals_region1(net,  data.x1, lamda))
    gpde1 = _residual_loss(gpde_residuals_region1(net, data.x1, lamda))
    pde2  = _residual_loss(pde_residuals_region2(net,  data.x2, lamda))
    gpde2 = _residual_loss(gpde_residuals_region2(net, data.x2, lamda))
    bc_r  = bc_loss_right(net, data.x_right)
    bc_l  = bc_loss_left (net, data.x_left)
    ic    = interface_loss(net, data.x_iface)

    return pde1 + gpde1 + pde2 + gpde2 + bc_r + bc_l + ic
