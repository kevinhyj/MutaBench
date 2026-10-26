# Reproducibility and release boundaries

The first release supports independent CSV evaluation, verification of selected
exported source/metadata, and redrawing five saved comparisons. It is a curated
export of the EVAgent research workspace with a new model-independent interface.
The existing workspace and frozen results were not rewritten.

## Included snapshots

`benchmark/source_manifest.json` records source-relative names, upstream hashes,
exported hashes and the transformation applied to each selected artifact.
Host paths and account metadata were removed from derived metric/catalog files;
their original scientific values and candidate order were preserved. Original
Python protocol/baseline source and publication images are copied byte-for-byte.
Legacy source can still contain historical path defaults; these are not runnable
public download locations or bundled credentials.

The relative `code` compatibility link points to `baselines/evagent` so original
protocol source-hash checks can locate the exported implementation. The small
`experiments/pg_rna_seed42_n250_v1/prepare.py` reference is the original cleaning
implementation used by synthetic sampling-replay tests; scientific data files
are not bundled there.

`scripts/verify_release.py` checks exported hashes, unique task and metric
identities, candidate counts, complete prediction counts and each plotted mean
against its per-assay values. This check is independent of the original project
directory. `scripts/plot_results.py` needs only the committed result packets.

## Comparison scope

The five current saved comparison cohorts are Protein50, RNA28, ADMET
classification13, ADMET regression8 and protein–protein27. They are not a
complete enumeration of all research modalities or experimental protocols.
Protein50 includes 48 N250 queries, one N200 query and one N224 query; these are
the complete original queries, with equal assay weights and no padding. The
other cohorts retain their saved membership and values.

Best-arm comparisons are descriptive same-cohort selection. The saved selection
rule fixes the complete assay cohort and chooses a complete arm/thinking
configuration for a model family by mean performance; it does not mix the best
arm separately for each assay. This is not a held-out estimate of model-selection
performance. Read each packet's `selection_rule` and `arm_candidates`.

The incomplete DeepSeek1017 pilot is excluded from the full-cohort comparison.
An older result synchronizer is not the authority for the merged saved metrics.
No new model scores were generated for this release.

## Scientific data and remaining boundaries

The separate Hugging Face release contains the selected public source data,
frozen zero-shot inputs and evaluator references, few-shot/multiround splits,
support examples, independent query evaluators and selected evidence assets.
The verified immutable commit and manifest checksum are bound in
`benchmark/data_release.json`; see [download and restore](data.md).

The data release is a curated benchmark export, not the entire research server.
Private cohorts, full API histories, raw prediction archives and some optional
upstream or historical assets are excluded or unavailable. `PLAN.json` records
scope and missing source paths. Recomputing metric values from raw predictions
is distinct from verifying saved metric packets. The Git checkout supports the
latter; it does not claim the former for every historical run.

External-model readiness varies. Historical optional weights, submodules,
services, licensed executables and host environments are not made complete by
publishing their adapter code. `baselines/catalog.json` is historical metadata,
not a guarantee of current service availability or all-assay inference coverage.
