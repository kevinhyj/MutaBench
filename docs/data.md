# Dataset distribution

The code repository contains metadata, hashes, synthetic examples and saved
performance tables. Full scientific inputs are a separate release. The intended
dataset destination is Hugging Face Datasets; upload status and an immutable
revision belong in `benchmark/data_release.json`. A destination name alone is
not evidence that a dataset has been uploaded or validated.

## Release layout

```text
MutaBench dataset/
  README.md                   dataset card and source-specific terms
  manifest.json               file sizes, SHA256, versions and provenance
  tasks/                      task definitions and readout/direction metadata
  queries/                    frozen identities and model-visible inputs
  support/                    measured examples allowed by each protocol
  evaluator/                  independent query labels
  evidence/                   MSA/structure/features, including missingness
  predictions/                selected saved predictions
```

Keep task/domain/source identity in every table. Parquet shards suit large
tabular data. Group structure/MSA files into documented archives when millions
of tiny files would make the repository unwieldy. Preserve original file hashes
inside a package manifest. Any archive conversion is a derived distribution
artifact and must not replace the hashes of frozen research inputs.

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

