# StyleGAN2 UniSub Analysis

This note keeps only the user-facing summary of the validated public StyleGAN2 release. For setup and rerun commands, use [README.md](./README.md).

## Validated Snapshot

- Source checkpoints: `20`
- Retained common `3x3` synthesis layers: `9`
- `q_all = 0.0572`
- `q_arch = 0.0907`

The retained layer list is committed in [retained_layers_conservative.csv](./retained_layers_conservative.csv), and the full release snapshot is committed in [experiment_snapshot.json](./experiment_snapshot.json).

## Quality Summary

The validated conservative run keeps the no-reference Inception Score essentially unchanged after reconstruction:

- IID reconstructed IS mean: `3.2042` with delta `-0.0041`
- OOD reconstructed IS mean: `3.2042` with delta `-0.0040`

These values come from [iid_ood_is_summary.csv](./iid_ood_is_summary.csv).

## Practical Takeaway

- Use the validated conservative run as the main public StyleGAN2 result.
- Treat lower-rank ablations as internal analysis rather than the primary release result.
- Use `README.md` for the full reproduction steps and this file only as a compact summary of what the release shows.
