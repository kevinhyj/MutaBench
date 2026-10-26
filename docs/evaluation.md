# Evaluating predictions

`mutabench evaluate` is an offline interface. It never calls a model. The task
manifest and labels must be the exact release artifacts used to generate a
prediction. The evaluator records SHA256 values of all three input files.

## Task manifest

```json
{
  "schema": "mutabench-tasks-v1",
  "tasks": [{
    "task_id": "source/assay/reference",
    "domain": "protein",
    "metric": "spearman",
    "higher_label_is_better": true,
    "candidate_ids": ["M001", "M002", "M003"]
  }]
}
```

Use a task ID that distinguishes source, assay and reference complex where
applicable. A bare mutation ID must not connect different assays. Candidate
order is frozen even though the CSV evaluator joins by identity.

`benchmark/singlechain_catalog.json` is a provenance/catalog format, not this
evaluation-input format. It contains hashes and candidate IDs, but no labels.
Task readout/direction and independently versioned labels must accompany the
future data release before scientific evaluation can run from a fresh checkout.

## Independent labels

```csv
task_id,variant_id,label
source/assay/reference,M001,0.1
source/assay/reference,M002,0.3
source/assay/reference,M003,0.2
```

Labels must cover exactly every manifest candidate. Nonfinite, missing,
unexpected or duplicate label identities fail validation. Query labels belong
to the evaluator, not the predictor. They must not be joined into prompts,
feature files, retrieval corpora or method-selection inputs.

## Predictions

```csv
task_id,variant_id,score,model,condition
source/assay/reference,M001,-0.2,my_model,sequence
source/assay/reference,M002,0.8,my_model,sequence
source/assay/reference,M003,0.3,my_model,sequence
```

`model` and `condition` are optional; defaults are `submitted` and `default`.
Each group is evaluated independently on every task in the manifest. A task
with no predictions is reported as incomplete, rather than silently dropped.
Unexpected candidates, duplicate identities and nonfinite scores fail validation.
Scores may use any finite numeric scale; historical LLM protocols separately
require scores in [-1, 1].

For Spearman, the declared label direction is applied before computing signed
correlation with average ranks for ties. Constant labels, constant scores and
tasks with fewer than two candidates have an undefined metric. AUROC requires
binary labels, uses average ranks for tied predictions, and is undefined when
only one label class is present. Constant scores yield AUROC 0.5 when both
classes are present.

The output includes per-task coverage/status and an equal-task-weight summary
for each model, condition, domain and metric. Spearman and AUROC are never
averaged together. Summary means may use different completed subsets; compare
the intersection of complete, defined tasks when comparing models. A null
metric is never replaced by zero.

