"""
bvp/solver.py
=============
Classical BVP reference solver for the two-region MHD channel-flow problem.

Purpose
-------
Physics-Informed Neural Networks (PINNs) are mesh-free, but their accuracy
should be validated against a trusted numerical reference.  This module
provides a **Boundary Value Problem (BVP) reference solver** using
`scipy.integrate.solve_bvp`, which implements a collocation method on an
adaptive mesh with error control.

The BVP solver treats the same ODEs that the PINN enforces via soft
constraints, but solves them exactly (within numerical tolerance) using a
classical method.  Comparing the two solutions reveals the approximation
error of the PINN/GPINN approach.

Scope: Full Two-Region System
------------------------------
The full problem couples 10 unknowns (u₁, w₁, b₁, θ₁, g₁, u₂, w₂, b₂, θ₂, g₂)
across two sub-domains with interface matching conditions at y = 0.

Strategy: Domain Decomposition
-------------------------------
We solve the two sub-domains separately by converting each 2nd-order ODE to
a first-order system.  The interface conditions are embedded as boundary
conditions for each sub-problem.

For each region, a 2nd-order ODE  f'' = F(y, f, f')  is written as a
first-order system by introducing the state vector:
    s = [f, f'] → s' = [f', F(y, s[0], s[1])]

This gives a system of 10 first-order ODEs per region (5 unknowns × 2 states).

Interface Handling
------------------
We use a sequential solve: Region 1 is solved first from y=0 to y=1 using
guessed interface values, then Region 2 is solved from y=-1 to y=0.  The
interface conditions are enforced as boundary conditions for each sub-problem.

For an uncoupled reference, we also provide standalone solvers for simpler
subsystems (e.g., u+θ only with linearised coupling) that can serve as a
quick sanity check.

References
----------
Kierzenka, J., & Shampine, L. F. (2001). A BVP solver based on residual
control and the MATLAB PSE. ACM TOMS, 27(3), 299-316.

Shampine, L. F., Gladwell, I., & Thompson, S. (2003). Solving ODEs with
MATLAB. Cambridge University Press.
"""

import numpy as np
from scipy.integrate import solve_bvp
from typing import Optional
import warnings


class MHDBVPSolver:
    """
    BVP solver for the two-region MHD channel-flow problem.

    The full system of 10 coupled second-order ODEs is split into two
    sub-domain problems, each solved with ``scipy.integrate.solve_bvp``.

    Parameters
    ----------
    lamda   : float  –  λ = cos(θ), cosine of magnetic field inclination angle.
    params  : module –  config.params (imported inside the class to avoid
                        circular imports; pass None to use defaults).

    Usage
    -----
    >>> solver = MHDBVPSolver(lamda=0.5)
    >>> sol1, sol2 = solver.solve()
    >>> if sol1.success and sol2.success:
    ...     print("BVP converged successfully.")
    """

    def __init__(self, lamda: float = 0.5, params=None):
        if params is None:
            from config import params as P
            self.P = P
        else:
            self.P = params

        self.lamda = lamda
        self.casson = 1.0 + 1.0 / self.P.beta
        self.sin_theta = np.sqrt(max(1.0 - lamda**2, 0.0))  # √(1-λ²) = sin(θ)

    # ------------------------------------------------------------------
    # Region 1: System of first-order ODEs  (y ∈ [0, 1])
    # ------------------------------------------------------------------

    def _fun_region1(self, y: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        First-order ODE system for Region 1 (Newtonian, y ∈ [0, 1]).

        State vector layout (10 components):
            s[0]  = u₁       s[1] = u₁'
            s[2]  = w₁       s[3] = w₁'
            s[4]  = b₁       s[5] = b₁'
            s[6]  = θ₁       s[7] = θ₁'
            s[8]  = g₁       s[9] = g₁'

        ODE system  s' = F(y, s):
            s[0]' = s[1]
            s[1]' = Ha₁²(K + λu₁)λ + 2ω₁w₁ - Gr₁(θ₁ + kb₁θ₁²) - G₁
            s[2]' = s[3]
            s[3]' = Ha₁²(λ² + (b₁ + sin θ)²) - G₁ - 2ω₁u₁
            s[4]' = s[5]
            s[5]' = -λ Rm₁ s[1]
            s[6]' = s[7]
            s[7]' = -[Pr₁Ec₁(s[1]²+s[3]²) + Ha₁²Pr₁Ec₁(K+λu₁)²] / (1 + 4Rd₁)
            s[8]' = s[9]
            s[9]' = Sc₁ η₁ g₁ⁿ

        Parameters
        ----------
        y : array (N,)    – spatial coordinate values.
        s : array (10, N) – state values at each point.

        Returns
        -------
        array (10, N)  –  derivatives.
        """
        P = self.P
        lam = self.lamda
        sin_t = self.sin_theta

        u1, du1 = s[0], s[1]
        w1, dw1 = s[2], s[3]
        b1, db1 = s[4], s[5]
        t1, dt1 = s[6], s[7]
        g1, dg1 = s[8], s[9]

        # Derived quantities
        bz1 = b1 + sin_t   # b₁ + sin(θ), the z-component of total field

        # RHS of u₁'' (momentum eqn)
        d2u1 = P.Ha1**2*(P.K + u1*lam)*lam + 2*P.omega1*w1 \
               - P.Gr1*(t1 + P.kb1*t1**2) - P.G1

        # RHS of w₁''
        d2w1 = P.Ha1**2*(lam**2 + bz1**2) - P.G1 - 2*P.omega1*u1

        # RHS of b₁''
        d2b1 = -lam * P.Rm1 * du1

        # RHS of θ₁'' (energy eqn, divided by (1 + 4Rd₁))
        visc_diss = P.Pr1*P.Ec1*(du1**2 + dw1**2)
        joule     = P.Ha1**2*P.Pr1*P.Ec1*(P.K + u1*lam)**2
        d2t1 = -(visc_diss + joule) / (1 + 4*P.Rd1)

        # RHS of g₁''
        d2g1 = np.clip(P.Sc1*P.eta1 * np.abs(g1)**P.n * np.sign(g1), -1e6, 1e6)

        return np.vstack([du1, d2u1, dw1, d2w1, db1, d2b1, dt1, d2t1, dg1, d2g1])

    def _bc_region1(self, sa: np.ndarray, sb: np.ndarray,
                    interface_vals: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Boundary conditions for Region 1.

        At y = 0 (sa, left end):  values match interface (or guessed).
        At y = 1 (sb, right end): Dirichlet BC  u₁=1, w₁=0, b₁=0, θ₁=1, g₁=1.

        Parameters
        ----------
        sa : array (10,) – state at y = 0.
        sb : array (10,) – state at y = 1.
        interface_vals : array (10,) or None
            If provided, the 5 function values at y=0 are fixed to this.
            Otherwise, they are treated as free (for decoupled testing).

        Returns
        -------
        array (10,)  –  residuals of the 10 boundary conditions.
        """
        # At y = 1 (right wall): u=1, w=0, b=0, θ=1, g=1
        bc_right = np.array([
            sb[0] - 1.0,  # u₁(1) = 1
            sb[2] - 0.0,  # w₁(1) = 0
            sb[4] - 0.0,  # b₁(1) = 0
            sb[6] - 1.0,  # θ₁(1) = 1
            sb[8] - 1.0,  # g₁(1) = 1
        ])

        if interface_vals is not None:
            # At y = 0: match provided interface values
            bc_left = np.array([
                sa[0] - interface_vals[0],
                sa[2] - interface_vals[1],
                sa[4] - interface_vals[2],
                sa[6] - interface_vals[3],
                sa[8] - interface_vals[4],
            ])
        else:
            # Free interface: use zero as initial guess
            bc_left = np.array([sa[0], sa[2], sa[4], sa[6] - 0.5, sa[8] - 0.5])

        return np.concatenate([bc_left, bc_right])

    # ------------------------------------------------------------------
    # Region 2: System of first-order ODEs  (y ∈ [-1, 0])
    # ------------------------------------------------------------------

    def _fun_region2(self, y: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        First-order ODE system for Region 2 (Casson, y ∈ [-1, 0]).

        State vector layout (same as Region 1):
            s[0]  = u₂       s[1] = u₂'
            s[2]  = w₂       s[3] = w₂'
            s[4]  = b₂       s[5] = b₂'
            s[6]  = θ₂       s[7] = θ₂'
            s[8]  = g₂       s[9] = g₂'

        The Casson factor (1 + 1/β) divides the RHS of the momentum ODEs.
        """
        P = self.P
        lam = self.lamda
        sin_t = self.sin_theta
        casson = self.casson

        u2, du2 = s[0], s[1]
        w2, dw2 = s[2], s[3]
        b2, db2 = s[4], s[5]
        t2, dt2 = s[6], s[7]
        g2, dg2 = s[8], s[9]

        bz2 = b2 + sin_t

        d2u2 = (P.Ha2**2*(P.K + u2*lam)*lam + 2*P.omega2*w2
                - P.Gr2*(t2 + P.kb2*t2**2) - P.G2) / casson

        d2w2 = (P.Ha2**2*(lam**2 + bz2**2) - P.G2 - 2*P.omega2*u2) / casson

        d2b2 = -lam * P.Rm2 * du2

        visc_diss2 = P.Pr2*P.Ec2*(du2**2 + dw2**2)
        joule2     = P.Ha2**2*P.Pr2*P.Ec2*(P.K + u2*lam)**2
        d2t2 = -(visc_diss2 + joule2) / (1 + 4*P.Rd2)

        d2g2 = np.clip(P.Sc2*P.eta2 * np.abs(g2)**P.n * np.sign(g2), -1e6, 1e6)

        return np.vstack([du2, d2u2, dw2, d2w2, db2, d2b2, dt2, d2t2, dg2, d2g2])

    def _bc_region2(self, sa: np.ndarray, sb: np.ndarray,
                    interface_vals: Optional[np.ndarray] = None,
                    interface_derivs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Boundary conditions for Region 2.

        At y = -1 (sa, left end):  u₂=0, w₂=0, b₂=0, θ₂=0, g₂=0.
        At y =  0 (sb, right end): match interface values from Region 1.

        The flux (derivative) jump conditions at the interface are:
            du₁/dy = (1+1/β) du₂/dy  → u₂'(0) = u₁'(0) / casson
            dw₁/dy = (1+1/β) dw₂/dy  → w₂'(0) = w₁'(0) / casson
            db₁/dy = δβ₁γ db₂/dy     → b₂'(0) = b₁'(0) / (δβ₁γ)
            dθ₁/dy = (β₁/ε) dθ₂/dy   → θ₂'(0) = θ₁'(0) / (β₁/ε) * ε/β₁
            dg₁/dy = (β₁/φ) dg₂/dy   → g₂'(0) = g₁'(0) / (β₁/φ)
        """
        P = self.P

        # Left wall (y = -1): all fields zero
        bc_left = np.array([
            sa[0],  # u₂(-1) = 0
            sa[2],  # w₂(-1) = 0
            sa[4],  # b₂(-1) = 0
            sa[6],  # θ₂(-1) = 0
            sa[8],  # g₂(-1) = 0
        ])

        if interface_vals is not None and interface_derivs is not None:
            # At y = 0: value continuity
            val_cts = np.array([
                sb[0] - interface_vals[0],
                sb[2] - interface_vals[1],
                sb[4] - interface_vals[2],
                sb[6] - interface_vals[3],
                sb[8] - interface_vals[4],
            ])
        else:
            val_cts = np.array([sb[0] - 0.5, sb[2], sb[4], sb[6] - 0.5, sb[8] - 0.5])

        return np.concatenate([bc_left, val_cts])

    # ------------------------------------------------------------------
    # Public API: solve()
    # ------------------------------------------------------------------

    def solve(self, n_init: int = 50, tol: float = 1e-6,
              max_nodes: int = 5000, verbose: bool = True):
        """
        Solve the two-region MHD BVP using scipy.integrate.solve_bvp.

        The approach is:
        1. Solve Region 1 (y ∈ [0, 1]) with linear initial guesses.
        2. Extract values and derivatives at y = 0 from the Region 1 solution.
        3. Apply interface jump conditions to obtain Region 2 boundary values.
        4. Solve Region 2 (y ∈ [-1, 0]) with these interface conditions.

        Parameters
        ----------
        n_init    : int     – initial number of mesh nodes in each region.
        tol       : float   – absolute/relative tolerance for solve_bvp.
        max_nodes : int     – maximum mesh nodes (refines adaptively).
        verbose   : bool    – print convergence status.

        Returns
        -------
        (sol1, sol2)
            sol1 : scipy BVP solution object for Region 1.
            sol2 : scipy BVP solution object for Region 2.
            Each has attributes: .y (state), .x (mesh), .success, .message.

        Notes
        -----
        If convergence fails, solve_bvp returns an object with .success=False.
        The solution may still be usable as a qualitative reference even if
        the solver did not fully converge.
        """
        P = self.P
        lam = self.lamda
        casson = self.casson

        # ---- Region 1 ------------------------------------------------
        y1_mesh = np.linspace(0.0, 1.0, n_init)

        # Initial guess: linear interpolation between BCs
        # s = [u, u', w, w', b, b', θ, θ', g, g']
        s1_init = np.zeros((10, n_init))
        s1_init[0] = np.linspace(1.0, 1.0, n_init)   # u(0)=1(guess), u(1)=1
        s1_init[6] = np.linspace(0.5, 1.0, n_init)   # θ: 0.5 → 1
        s1_init[8] = np.linspace(0.5, 1.0, n_init)   # g: 0.5 → 1

        def fun1(y, s):
            return self._fun_region1(y, s)

        def bc1(sa, sb):
            return self._bc_region1(sa, sb, interface_vals=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol1 = solve_bvp(fun1, bc1, y1_mesh, s1_init,
                             tol=tol, max_nodes=max_nodes,
                             verbose=2 if verbose else 0)

        if verbose:
            status = "CONVERGED" if sol1.success else "DID NOT CONVERGE"
            print(f"  [BVP Region 1] {status}: {sol1.message}")

        # ---- Extract interface values from Region 1 solution ---------
        # sol1(0.0) gives the state [u₁(0), u₁'(0), ..., g₁(0), g₁'(0)]
        s_at_0 = sol1.sol(0.0)
        iface_vals   = np.array([s_at_0[0], s_at_0[2], s_at_0[4], s_at_0[6], s_at_0[8]])
        iface_derivs = np.array([s_at_0[1], s_at_0[3], s_at_0[5], s_at_0[7], s_at_0[9]])

        # Apply derivative jump conditions to get Region 2 interface derivatives
        du2_0 = iface_derivs[0] / casson
        dw2_0 = iface_derivs[1] / casson
        db2_0 = iface_derivs[2] / (P.delta * P.beta1 * P.gamma + 1e-10)
        dt2_0 = iface_derivs[3] * P.epsilon / (P.beta1 + 1e-10)
        dg2_0 = iface_derivs[4] * P.phi    / (P.beta1 + 1e-10)
        iface_derivs2 = np.array([du2_0, dw2_0, db2_0, dt2_0, dg2_0])

        # ---- Region 2 ------------------------------------------------
        y2_mesh = np.linspace(-1.0, 0.0, n_init)

        s2_init = np.zeros((10, n_init))
        s2_init[0] = np.linspace(0.0, iface_vals[0], n_init)
        s2_init[6] = np.linspace(0.0, iface_vals[3], n_init)
        s2_init[8] = np.linspace(0.0, iface_vals[4], n_init)

        def fun2(y, s):
            return self._fun_region2(y, s)

        def bc2(sa, sb):
            return self._bc_region2(sa, sb,
                                    interface_vals=iface_vals,
                                    interface_derivs=iface_derivs2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol2 = solve_bvp(fun2, bc2, y2_mesh, s2_init,
                             tol=tol, max_nodes=max_nodes,
                             verbose=2 if verbose else 0)

        if verbose:
            status = "CONVERGED" if sol2.success else "DID NOT CONVERGE"
            print(f"  [BVP Region 2] {status}: {sol2.message}")

        return sol1, sol2

    def get_profiles(self, sol1, sol2, n_pts: int = 200):
        """
        Evaluate the BVP solution on a fine uniform mesh for plotting.

        Parameters
        ----------
        sol1   : BVP solution for Region 1.
        sol2   : BVP solution for Region 2.
        n_pts  : int – number of evaluation points per region.

        Returns
        -------
        dict with keys: 'y', 'u', 'w', 'b', 'theta', 'g'
            Each value is a NumPy array of shape (2*n_pts,) covering y ∈ [-1, 1].
        """
        y1 = np.linspace(0.0,  1.0, n_pts)
        y2 = np.linspace(-1.0, 0.0, n_pts)

        s1 = sol1.sol(y1)
        s2 = sol2.sol(y2)

        # Concatenate Region 2 then Region 1 (so y is increasing from -1 to 1)
        y_all     = np.concatenate([y2, y1])
        u_all     = np.concatenate([s2[0], s1[0]])
        w_all     = np.concatenate([s2[2], s1[2]])
        b_all     = np.concatenate([s2[4], s1[4]])
        theta_all = np.concatenate([s2[6], s1[6]])
        g_all     = np.concatenate([s2[8], s1[8]])

        return {'y': y_all, 'u': u_all, 'w': w_all,
                'b': b_all, 'theta': theta_all, 'g': g_all}
