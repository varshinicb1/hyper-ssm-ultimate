# Legacy / Archived Code

This folder contains older experimental architectures that are no longer the main focus of the project.

## Contents (as of 2026 Ultimate refresh)

- `model.py` — Original ALR (Adaptive Latent Recurrence) model
- `train.py` — Training script for ALR
- `loss.py` — OrthogonalKnowledgeLoss used with ALR
- `inference.py` — Old inference demo for ALR

These were early explorations before the project converged on the Lorentzian Hyper-SSM + Liquid Experts direction.

**Recommendation**: Do not use for new work. The active code lives in `hyper_ssm/`.

See `HYPER_SSM_2026_ULTIMATE.md` for the current integrated vision.
