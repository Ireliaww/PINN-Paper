"""
pinn/trainer.py
===============
Training orchestration for the PINN and GPINN solvers.

Training Strategy
-----------------
A two-phase optimisation strategy is used, following common practice in the
PINN literature:

  Phase 1 – Adam optimiser (epochs = ADAM_EPOCHS):
    Adam (Adaptive Moment Estimation) is a first-order gradient method with
    adaptive per-parameter learning rates.  It is robust to poor initialisations
    and converges quickly in the early training phase, moving the network from
    a random state to the vicinity of a good solution.

  Phase 2 – L-BFGS optimiser (max_iter = LBFGS_MAX_ITER):
    L-BFGS is a quasi-Newton second-order method that exploits curvature
    information of the loss landscape.  Once Adam has found a reasonable basin
    of attraction, L-BFGS typically achieves much lower loss values and tighter
    PDE satisfaction than first-order methods alone.
    It uses a full-batch closure (re-evaluates the loss on each line-search step).

ThresholdTracker
----------------
Records the cumulative training epoch and wall-clock time (seconds) at which
the total loss first drops below each of the user-specified thresholds:
    [1e-4, 1e-6, 1e-8]

This enables a quantitative comparison of PINN vs GPINN convergence speed.

References
----------
Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large
scale optimization. Mathematical programming, 45(1-3), 503-528.

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980.
"""

import time
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from config import params as P


# ---------------------------------------------------------------------------
# ThresholdTracker
# ---------------------------------------------------------------------------

class ThresholdTracker:
    """
    Tracks the first time the total training loss falls below each threshold.

    Attributes
    ----------
    thresholds : list of float
        Loss values at which to record timing/epoch data.
    epochs : dict[float, int | None]
        The cumulative epoch number when the threshold was first crossed
        (None if not yet reached).
    times : dict[float, float | None]
        Wall-clock time in seconds since training start when the threshold
        was first crossed (None if not yet reached).
    """

    def __init__(self, thresholds: List[float]):
        self.thresholds = thresholds
        self.epochs: Dict[float, Optional[int]]   = {t: None for t in thresholds}
        self.times:  Dict[float, Optional[float]] = {t: None for t in thresholds}

    def update(self, loss_value: float, cumulative_epoch: int, t0: float) -> None:
        """
        Check whether the current loss crosses any unrecorded threshold.

        Parameters
        ----------
        loss_value        : float  –  current total loss (scalar).
        cumulative_epoch  : int    –  total epochs completed across both phases.
        t0                : float  –  training start timestamp (time.time()).
        """
        for thr in self.thresholds:
            if self.epochs[thr] is None and loss_value <= thr:
                self.epochs[thr] = cumulative_epoch
                self.times[thr]  = time.time() - t0

    def summary(self) -> str:
        """Return a formatted string summarising the threshold crossing data."""
        lines = []
        for thr in self.thresholds:
            ep = self.epochs[thr]
            tm = self.times[thr]
            ep_str = f"{ep:>6d}" if ep is not None else "   N/A"
            tm_str = f"{tm:>8.2f}s" if tm is not None else "     N/A"
            lines.append(f"  Threshold {thr:.0e}: epoch {ep_str}, time {tm_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training Phases
# ---------------------------------------------------------------------------

def train_adam(
    net: nn.Module,
    loss_fn: Callable,
    data,
    lamda: float,
    tracker: ThresholdTracker,
    t0: float,
    epoch_offset: int = 0,
    print_every: int  = 100,
) -> List[float]:
    """
    Phase 1 training: Adam optimiser.

    Runs ADAM_EPOCHS iterations of Adam with learning rate LR_ADAM.
    At each epoch the total loss is computed, back-propagated, and the
    network weights are updated.

    Parameters
    ----------
    net          : nn.Module         – PINNNet or GPINNNet.
    loss_fn      : callable          – total_loss_pinn or total_loss_gpinn.
    data         : CollocationData   – collocation tensors.
    lamda        : float             – λ = cos(θ).
    tracker      : ThresholdTracker  – records first-crossing epochs/times.
    t0           : float             – wall-clock start time.
    epoch_offset : int               – cumulative epochs already done (0 for Phase 1).
    print_every  : int               – print loss every N epochs.

    Returns
    -------
    List[float]  –  per-epoch loss values.
    """
    optimizer = optim.Adam(net.parameters(), lr=P.LR_ADAM)
    loss_history = []

    for epoch in range(1, P.ADAM_EPOCHS + 1):
        optimizer.zero_grad()
        loss = loss_fn(net, data, lamda)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        cum_epoch = epoch_offset + epoch
        tracker.update(loss_val, cum_epoch, t0)

        if epoch % print_every == 0:
            print(f"  [Adam epoch {epoch:>4d}/{P.ADAM_EPOCHS}]  "
                  f"loss = {loss_val:.4e}  "
                  f"(net: {net.__class__.__name__})")

    return loss_history


def train_lbfgs(
    net: nn.Module,
    loss_fn: Callable,
    data,
    lamda: float,
    tracker: ThresholdTracker,
    t0: float,
    epoch_offset: int = 0,
    print_every: int  = 100,
) -> List[float]:
    """
    Phase 2 training: L-BFGS optimiser.

    L-BFGS requires a closure function that re-evaluates the loss and its
    gradient on demand (needed by the line-search algorithm inside L-BFGS).

    The `tolerance_grad` and `tolerance_change` are set very tight (1e-9)
    to allow L-BFGS to run for the full `LBFGS_MAX_ITER` budget rather than
    stopping early due to loose convergence criteria.

    Parameters
    ----------
    net, loss_fn, data, lamda, tracker, t0 : see train_adam.
    epoch_offset : int  –  P.ADAM_EPOCHS (so cumulative epoch count is correct).
    print_every  : int  –  print loss every N L-BFGS steps.

    Returns
    -------
    List[float]  –  per-step loss values.
    """
    optimizer = optim.LBFGS(
        net.parameters(),
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        history_size=100,
    )
    loss_history = []

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(net, data, lamda)
        loss.backward()
        return loss

    for step in range(1, P.LBFGS_MAX_ITER + 1):
        loss = optimizer.step(closure)
        loss_val = loss.item()
        loss_history.append(loss_val)

        cum_epoch = epoch_offset + step
        tracker.update(loss_val, cum_epoch, t0)

        if step % print_every == 0:
            print(f"  [L-BFGS step {step:>5d}/{P.LBFGS_MAX_ITER}]  "
                  f"loss = {loss_val:.4e}  "
                  f"(net: {net.__class__.__name__})")

    return loss_history


# ---------------------------------------------------------------------------
# High-Level Training Orchestrator
# ---------------------------------------------------------------------------

def train_network(
    net: nn.Module,
    loss_fn: Callable,
    data,
    lamda: float,
    save_path: str,
    label: str = "Net",
) -> Tuple[List[float], ThresholdTracker]:
    """
    Orchestrate the full two-phase (Adam → L-BFGS) training pipeline.

    Steps
    -----
    1. Initialise ThresholdTracker and start the global timer.
    2. Run Phase 1 (Adam) for P.ADAM_EPOCHS epochs.
    3. Run Phase 2 (L-BFGS) for P.LBFGS_MAX_ITER steps.
    4. Save the trained model weights to `save_path`.
    5. Return the combined loss history and tracker.

    Parameters
    ----------
    net       : nn.Module         – network to train.
    loss_fn   : callable          – total_loss_pinn or total_loss_gpinn.
    data      : CollocationData   – collocation tensors.
    lamda     : float             – λ = cos(θ).
    save_path : str               – filepath for saving trained weights (.pt).
    label     : str               – display name for logging.

    Returns
    -------
    (loss_history, tracker)
        loss_history : List[float]      – all per-epoch losses (Adam + L-BFGS).
        tracker      : ThresholdTracker – convergence data.
    """
    tracker = ThresholdTracker(P.LOSS_THRESHOLDS)
    t0      = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {label}  |  λ = {lamda}")
    print(f"  Phase 1: Adam ({P.ADAM_EPOCHS} epochs, lr={P.LR_ADAM})")
    print(f"{'='*60}")
    adam_history = train_adam(
        net, loss_fn, data, lamda, tracker, t0,
        epoch_offset=0,
    )

    print(f"\n  Phase 2: L-BFGS ({P.LBFGS_MAX_ITER} steps)")
    print(f"{'='*60}")
    lbfgs_history = train_lbfgs(
        net, loss_fn, data, lamda, tracker, t0,
        epoch_offset=P.ADAM_EPOCHS,
    )

    # Save trained model weights to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"\n  Model saved → {save_path}")

    print(f"\n  Convergence summary for {label}:")
    print(tracker.summary())

    loss_history = adam_history + lbfgs_history
    return loss_history, tracker
