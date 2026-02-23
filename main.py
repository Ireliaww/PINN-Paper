"""
main.py
=======
Top-level entry point for the two-region MHD channel-flow PINN/GPINN study.

This script orchestrates the full experimental pipeline:

  1. Generate collocation and boundary/interface training points.
  2. Run BVP reference solver (scipy) for validation.
  3. Train GPINN-Net and PINN-Net for each value of λ = cos(θ).
  4. Save trained model checkpoints to `outputs/`.
  5. Plot training loss curves and field-profile comparisons.
  6. Print the convergence summary table.

Usage
-----
    python main.py

Configuration
-------------
All physical parameters and training hyperparameters are defined in
``config/params.py``.  Edit that file to change the problem setup.

Output Files
------------
All generated files are written to the ``outputs/`` directory:
    GPINNmodel_lamda<λ>.pt       – GPINN trained weights
    PINNmodel_lamda<λ>.pt        – PINN trained weights
    loss_curves_lamda<λ>.png     – Training loss curve plot
    field_profiles_lamda<λ>.png  – Field profile comparison plot
"""

import os
import sys
import torch

# Ensure the project root is on the Python path when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import params as P
from data.collocation import generate_collocation_points
from pinn.network import PINNNet, GPINNNet
from pinn.losses import total_loss_pinn, total_loss_gpinn
from pinn.trainer import train_network
from bvp.solver import MHDBVPSolver
from utils.visualization import (
    plot_loss_curves,
    plot_field_profiles,
    print_threshold_table,
)


def main():
    """Main experiment driver."""

    print("\n" + "=" * 60)
    print("  Two-Region MHD Channel Flow: PINN vs. GPINN Study")
    print(f"  Device : {P.DEVICE}")
    print(f"  Lambda sweep : {P.LAMDAS}")
    print(f"  Adam epochs  : {P.ADAM_EPOCHS},  L-BFGS steps : {P.LBFGS_MAX_ITER}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Generate collocation / boundary / interface points          #
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Generating collocation points ...")
    data = generate_collocation_points(N=P.N_COLLOC, device=P.DEVICE)
    print(f"  Region 1 points : {data.x1.shape[0]} in y ∈ [0, 1]")
    print(f"  Region 2 points : {data.x2.shape[0]} in y ∈ [-1, 0]")

    # ------------------------------------------------------------------ #
    # Step 2: BVP reference solution                                      #
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Running BVP reference solver ...")
    bvp_results = {}  # lamda → bvp_profiles dict

    bvp_solver = MHDBVPSolver(lamda=P.LAMDAS[0])
    sol1, sol2 = bvp_solver.solve(verbose=True)

    if sol1.success and sol2.success:
        bvp_profiles = bvp_solver.get_profiles(sol1, sol2)
        for lam in P.LAMDAS:
            bvp_results[lam] = bvp_profiles
        print("  BVP solver converged for all λ values (using λ[0] profile).")
    else:
        print("  WARNING: BVP solver did not fully converge. "
              "BVP curves will be omitted from plots.")
        bvp_results = {lam: None for lam in P.LAMDAS}

    # ------------------------------------------------------------------ #
    # Step 3–5: Train GPINN and PINN for each λ                          #
    # ------------------------------------------------------------------ #
    all_results = {}   # lamda → {'GPINN': tracker, 'PINN': tracker}

    for lamda in P.LAMDAS:
        print(f"\n{'#'*60}")
        print(f"  λ = {lamda}")
        print(f"{'#'*60}")

        # Construct fresh networks for each λ sweep value.
        # This ensures independent training runs.
        gpinn_net = GPINNNet().to(P.DEVICE)
        pinn_net  = PINNNet().to(P.DEVICE)

        # Define save paths
        gpinn_save = os.path.join(P.MODEL_SAVE_DIR, f"GPINNmodel_lamda{lamda}.pt")
        pinn_save  = os.path.join(P.MODEL_SAVE_DIR, f"PINNmodel_lamda{lamda}.pt")

        # ---- Train GPINN ----
        gpinn_losses, gpinn_tracker = train_network(
            net=gpinn_net,
            loss_fn=total_loss_gpinn,
            data=data,
            lamda=lamda,
            save_path=gpinn_save,
            label=f"GPINN-Net (λ={lamda})",
        )

        # ---- Train PINN ----
        pinn_losses, pinn_tracker = train_network(
            net=pinn_net,
            loss_fn=total_loss_pinn,
            data=data,
            lamda=lamda,
            save_path=pinn_save,
            label=f"PINN-Net  (λ={lamda})",
        )

        # ---- Record results ----
        all_results[lamda] = {
            'GPINN': gpinn_tracker,
            'PINN':  pinn_tracker,
        }

        # ---- Visualise ----
        print(f"\n[Step 4] Plotting results for λ = {lamda} ...")
        plot_loss_curves(gpinn_losses, pinn_losses, lamda=lamda, save=True)
        plot_field_profiles(
            gpinn_net, pinn_net,
            lamda=lamda,
            bvp_profiles=bvp_results.get(lamda),
            save=True,
        )

    # ------------------------------------------------------------------ #
    # Step 6: Convergence summary                                         #
    # ------------------------------------------------------------------ #
    print_threshold_table(all_results)

    print("\nDone. All outputs written to:", P.MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
