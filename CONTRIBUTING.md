# Contribute a dataset

Labs can propose a measured assay or a collection of related assays. Start with the
[dataset proposal form](https://github.com/kevinhyj/MutaBench/issues/new?template=dataset.yml)
or open a pull request directly if your public development sample is ready.
Include the source citation, who contributed the data, the measured endpoint and
what MutaBench may distribute. Only share material cleared for the repository.

## Prepare a public development sample

Copy the [working example](examples/community/example-fluorescence/) to
`community/<dataset-id>/`. Replace its invented records and metadata with
your own. The sample contains four files.

- **dataset.json** records a stable ID, version, contributors, source, source revision, license or access terms, experimental readout and split policy. Describe units, direction, replicate handling and overlap with earlier releases. For a large public dataset, give its permanent source URL or DOI and pin the exact revision.
- **tasks.json** defines one task per assay or endpoint. Prefix its ID with the dataset ID and list the frozen candidate IDs, domain, metric and label direction. The [evaluation format](docs/evaluation.md) defines Spearman and AUROC.
- **queries.csv** contains `task_id,variant_id` and the model-visible input fields. For a complex, identify the molecular partners. For variant notation, document the reference sequence and numbering convention in the readout or source description.
- **labels.csv** contains `task_id,variant_id,label` for the public development sample. Match identities across files. Row order is free.

Set `test_access` to `development-only` for public-only contributions or
`withheld` when separate test labels are available for controlled evaluation.
For withheld data, describe the proposed access terms in the issue and arrange
private transfer with a maintainer. Keep hidden inputs, labels, access links and
credentials outside the PR, its history and Actions logs. A source whose labels
are already publicly retrievable should be identified as such in the split description.

## Check and submit

From the repository root with the package installed, run the example:

```sh
python -m mutabench validate-dataset examples/community/example-fluorescence
```

Run the same command on your contribution directory, then open a PR. The
validator checks the package layout, task definitions, candidate identities,
label domains and hashes. Its report includes candidate and distinct-label
counts per assay. The existing evaluation CLI accepts the development tasks
and labels directly, along with a model's predictions.

The PR checks all contributed development samples with the same validator.
A maintainer reviews readout interpretation, data rights, duplicates and overlap
with existing tasks. Automated checks only recognize common target column names.
Review the meaning of every model-visible input, including IDs and filenames.

## Release and credit

A merged sample appears in the [community catalog](community/). The benchmark
maintainers choose the tasks for each tagged evaluation release and cite the
contributors and original data source. A community PR alone does not produce an
official score. For a withheld evaluation, maintainers freeze the allowed inputs,
split and evaluator version on the assessment server and return approved aggregate
metrics. Data access details and individual hidden measurements stay there.

For a correction, bump the version and describe its impact in the PR. Record
whether the change requires new predictions, rescoring saved predictions, or
only a metadata update. Previous releases remain addressable by their Git tag
or commit.
