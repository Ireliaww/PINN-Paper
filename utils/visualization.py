"""
utils/visualization.py
=======================
Plotting and reporting utilities for the PINN/GPINN MHD flow study.

All figures are saved to the `outputs/` directory as PNG files.

Functions
---------
plot_loss_curves       – training loss vs. epoch (log scale) for PINN and GPINN.
plot_field_profiles    – spatial profiles of each field variable compared
                         across PINN, GPINN, and BVP reference.
print_threshold_table  – formatted console table of convergence statistics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from typing import Optional

from config import params as P

# Default figure output directory
_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(_OUT_DIR, exist_ok=True)

# ─── Matplotlib style settings ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
})

# Field variable display metadata (name, LaTeX label, column indices)
_FIELDS = [
    ("u",     r"$u(y)$",     0, 5),   # axial velocity
    ("w",     r"$w(y)$",     1, 6),   # transverse velocity
    ("b",     r"$b(y)$",     2, 7),   # induced magnetic field
    ("theta", r"$\theta(y)$",3, 8),   # temperature
    ("g",     r"$g(y)$",     4, 9),   # species concentration
]


# ---------------------------------------------------------------------------
# Loss Curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    gpinn_losses: list,
    pinn_losses:  list,
    lamda: float,
    save: bool = True,
) -> plt.Figure:
    """
    Plot training loss histories for GPINN and PINN on a logarithmic y-axis.

    A vertical dashed line is drawn at epoch ADAM_EPOCHS to indicate the
    Adam → L-BFGS phase transition.

    Parameters
    ----------
    gpinn_losses : list of float  –  per-epoch GPINN total loss.
    pinn_losses  : list of float  –  per-epoch PINN total loss.
    lamda        : float          –  λ value used (for the title and filename).
    save         : bool           –  whether to write the figure to disk.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    epochs_gpinn = range(1, len(gpinn_losses) + 1)
    epochs_pinn  = range(1, len(pinn_losses)  + 1)

    ax.semilogy(epochs_gpinn, gpinn_losses,
                label="GPINN-Net", linewidth=2,
                marker='o', markersize=3, markevery=max(len(gpinn_losses)//20, 1),
                alpha=0.85)
    ax.semilogy(epochs_pinn, pinn_losses,
                label="PINN-Net",  linewidth=2,
                marker='s', markersize=3, markevery=max(len(pinn_losses)//20, 1),
                alpha=0.85)

    # Mark the Adam → L-BFGS transition
    ax.axvline(x=P.ADAM_EPOCHS, color='gray', linestyle='--',
               linewidth=1.2, label=f"Adam → L-BFGS (epoch {P.ADAM_EPOCHS})")

    ax.set_xlabel("Cumulative Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title(f"Training Loss Curves  (λ = {lamda})")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_xlim(left=0)
    fig.tight_layout()

    if save:
        path = os.path.join(_OUT_DIR, f"loss_curves_lamda{lamda}.png")
        fig.savefig(path, dpi=150)
        print(f"  [Plot] Loss curves saved → {path}")

    return fig


# ---------------------------------------------------------------------------
# Field Profile Comparison
# ---------------------------------------------------------------------------

def plot_field_profiles(
    gpinn_net,
    pinn_net,
    lamda: float,
    bvp_profiles: Optional[dict] = None,
    save: bool = True,
) -> plt.Figure:
    """
    Plot spatial profiles of all five field variables, comparing GPINN, PINN, and BVP.

    For each field f ∈ {u, w, b, θ, g}:
      - Network values on y ∈ [0, 1]  use output indices 0–4 (Region 1).
      - Network values on y ∈ [-1, 0] use output indices 5–9 (Region 2).

    Parameters
    ----------
    gpinn_net    : GPINNNet  –  trained GPINN network.
    pinn_net     : PINNNet   –  trained PINN network.
    lamda        : float     –  λ value (for the title/filename).
    bvp_profiles : dict or None
        Dictionary with keys 'y', 'u', 'w', 'b', 'theta', 'g' (from
        MHDBVPSolver.get_profiles).  If None, BVP curves are omitted.
    save         : bool      –  whether to write the figure to disk.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Evaluation grids
    y_r1 = torch.linspace(0.0,  1.0, 200)[:, None]  # Region 1
    y_r2 = torch.linspace(-1.0, 0.0, 200)[:, None]  # Region 2
    y_full = torch.cat([y_r2, y_r1], dim=0)          # full domain

    with torch.no_grad():
        out_gpinn_r1 = gpinn_net(y_r1).numpy()
        out_gpinn_r2 = gpinn_net(y_r2).numpy()
        out_pinn_r1  = pinn_net(y_r1).numpy()
        out_pinn_r2  = pinn_net(y_r2).numpy()

    y1 = y_r1.numpy().flatten()
    y2 = y_r2.numpy().flatten()

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    field_keys = ['u', 'w', 'b', 'theta', 'g']

    for ax, (fname, flabel, idx1, idx2) in zip(axes, _FIELDS):
        # GPINN: stitch Region 2 and Region 1
        gpinn_vals = np.concatenate([out_gpinn_r2[:, idx2], out_gpinn_r1[:, idx1]])
        # PINN
        pinn_vals  = np.concatenate([out_pinn_r2[:, idx2],  out_pinn_r1[:, idx1]])
        y_plot     = np.concatenate([y2, y1])

        ax.plot(gpinn_vals, y_plot, '-',  label="GPINN", linewidth=2)
        ax.plot(pinn_vals,  y_plot, '--', label="PINN",  linewidth=2)

        if bvp_profiles is not None and fname in bvp_profiles:
            ax.plot(bvp_profiles[fname], bvp_profiles['y'],
                    ':', label="BVP ref.", linewidth=2, color='k')

        ax.set_xlabel(flabel)
        ax.set_ylabel(r"$y$")
        ax.set_title(fname)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.axhline(0, color='k', linewidth=0.6)

    fig.suptitle(f"Field Profiles – PINN vs. GPINN vs. BVP  (λ = {lamda})",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    if save:
        path = os.path.join(_OUT_DIR, f"field_profiles_lamda{lamda}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Field profiles saved → {path}")

    return fig


# ---------------------------------------------------------------------------
# Threshold Convergence Table
# ---------------------------------------------------------------------------

def print_threshold_table(results: dict) -> None:
    """
    Print a formatted table summarising convergence statistics.

    The table shows, for each λ value and each loss threshold, the epoch and
    wall-clock time at which GPINN-Net and PINN-Net first reached that threshold.
    A value of 'N/A' means the threshold was not reached within the training budget.

    Parameters
    ----------
    results : dict
        Structured as:
        {
          lamda_value: {
            'GPINN': ThresholdTracker,
            'PINN':  ThresholdTracker,
          },
          ...
        }
    """
    header_width = 80
    print("\n" + "=" * header_width)
    print("  CONVERGENCE SUMMARY TABLE")
    print("=" * header_width)
    print(f"  {'Lambda':>8}  {'Threshold':>10}  "
          f"{'GPINN Epoch':>12}  {'GPINN Time(s)':>14}  "
          f"{'PINN Epoch':>11}  {'PINN Time(s)':>13}")
    print("-" * header_width)

    for lamda, res in results.items():
        gpinn_t = res['GPINN']
        pinn_t  = res['PINN']
        for thr in P.LOSS_THRESHOLDS:
            ge = gpinn_t.epochs.get(thr)
            gt = gpinn_t.times.get(thr)
            pe = pinn_t.epochs.get(thr)
            pt = pinn_t.times.get(thr)

            ge_s = f"{ge:>12d}"    if ge is not None else "         N/A"
            gt_s = f"{gt:>14.2f}" if gt is not None else "           N/A"
            pe_s = f"{pe:>11d}"    if pe is not None else "        N/A"
            pt_s = f"{pt:>13.2f}" if pt is not None else "          N/A"

            print(f"  {lamda:>8.3f}  {thr:>10.0e}  {ge_s}  {gt_s}  {pe_s}  {pt_s}")
        print("-" * header_width)

    print("=" * header_width + "\n")
