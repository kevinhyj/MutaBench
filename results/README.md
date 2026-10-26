# Saved results

`single_chain_metrics.json` contains 2,828 complete saved single-chain and ADMET
metric cells. The curated export removes account names, route metadata and local
source paths while preserving scientific identity, score, coverage and query
hash. A complete cell may still have an undefined metric, represented by null.

`complex_metrics.json` contains 1,792 saved protein–protein, protein–RNA,
protein–DNA and protein–ligand cells. Failed and incomplete statuses remain
explicit; they are not filled from superseded runs. Current assay exclusions
are recorded in `benchmark/evaluation_exclusions.json`.

`figure_packets/` contains exact cohort membership, bar configuration, per-assay
values, means and selection policies for five current historical comparisons.
Best-arm selection is descriptive and uses the displayed cohort; it is not
held-out validation. Some figure baselines have different availability from the
full registered source pool.

Run `python scripts/verify_release.py` to verify hashes and packet consistency,
and `python scripts/plot_results.py --output outputs/figures` to redraw the saved
values. Raw prediction and label files will be distributed separately; this
checkout alone cannot recompute every original metric from raw scores.
