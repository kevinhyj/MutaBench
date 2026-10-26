# Dataset distribution

The code repository contains metadata, hashes, synthetic examples and saved
performance tables. Scientific inputs are distributed separately in
[yanjiehuang/MutaBench](https://huggingface.co/datasets/yanjiehuang/MutaBench).
`benchmark/data_release.json` pins the verified dataset commit and release
manifest checksum. See that binding for visibility and upload verification.

## Release layout

```text
MutaBench dataset/
  README.md                   dataset card and loading configurations
  SOURCES.md                  upstream sources and source-specific terms
  RELEASE.json                archive/table sizes and SHA256 checksums
  FILES.jsonl                 original file paths, sizes, hashes and archive
  PLAN.json                   export scope, missing assets and exclusions
  tables/queries.parquet      label-free single-chain queries
  tables/labels.parquet       independent evaluator labels
  tables/labels.csv           labels in the evaluation CLI format
  tables/tasks.json           task definitions for the evaluation CLI
  archives/data-0000.tar.gz   original files at pipeline-relative paths
  ...
  archives/data-0058.tar.gz
```

The 59 archives contain 217,835 files: public single-chain and complex sources,
ADMET inputs, zero-shot inputs and evaluator references, few-shot/multiround
splits, allowed support examples, independent query evaluators, and selected
MSA, structure and feature evidence. `FILES.jsonl` records each original file's
hash and archive membership. Restoration preserves the original paths and bytes.

The 249-task query/label tables cover the registered single-chain source pool;
they are not a unified table of every complex, ADMET or few-shot protocol.
Use the corresponding frozen archive files and protocol for those experiments.
Historical versions and selected figure cohorts remain distinct. Complete raw
prediction archives and API call histories are not included, so this release
does not support recomputing every historical metric from raw predictions.

## Versioning and boundaries

- Pin the dataset commit revision in code release metadata; do not rely on a
  mutable `main` revision to reproduce a paper.
- Record upstream version, source location, readout, units, score direction,
  cleaning/deduplication and candidate-selection rules for every assay.
- Separate the registered source pool from eligibility, available evidence,
  admitted model tasks, complete predictions and each reported paired cohort.
- Keep support and independent query labels in separate files and loading paths.
- Preserve missing features as unavailable/null with reasons. Missing evidence
  is not a score of zero.
- Include only assets cleared for redistribution under their source terms.
  Private cohorts and sources with unresolved sharing permission are not part
  of the public package by default. This also applies to prompts or prediction
  records containing their sequences or measurements.

Model weights and third-party executables belong to their original distribution
channels. The benchmark should point to their exact version/checksum and setup
instructions rather than duplicate installed environments or proprietary tools.

## Download and restore

**Access:** the dataset repository is currently private. Log in with an account
that has access before downloading. Publishing this code does not grant dataset
access or change its visibility.

Install the data extra and download either the lightweight tables or the complete pack:

```bash
python -m pip install -e '.[data]'
python scripts/download_data.py --tables-only --output outputs/data-tables

# Full release, into a different fresh directory:
python scripts/download_data.py --output outputs/data-full
python3.12 scripts/restore_data.py \
  --download outputs/data-full --output /path/to/restored-data
```

Authentication uses your normal Hugging Face login or `HF_TOKEN`; credentials
are not stored in this repository. A private dataset requires access through
its owning account or organization. The downloader pins an immutable revision
and verifies artifact SHA256 values. The restorer additionally validates tar
membership and each restored file's byte size and SHA256; external symbolic
links and archive traversal are rejected.

The tables provide 61,688 frozen, label-free queries across the 249 registered
single-chain tasks, matching independent evaluator labels and `tasks.json` for
the model-independent evaluation CLI. Few-shot/multiround splits, support,
independent evaluators and evidence remain in the pipeline-relative archives.
The release pack is about 24.6 GB compressed and 54.5 GB expanded. Keep these
separate from the runtime dependencies and optional model weights.

Historical input versions remain frozen for provenance. Use the intended
version and evaluation exclusions; older prompts or assay scopes must not be
treated as interchangeable with the current comparison.
