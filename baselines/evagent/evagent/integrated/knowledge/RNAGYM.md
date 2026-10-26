# RNAgym evaluation knowledge

This agent evaluates zero-shot RNA fitness prediction using the existing `RG-LLM` artifacts.

- Protocol: `ranking-rna-v1`.
- Episode: one assay, 50 sampled mutants, seed 1.
- Output: strict JSON object containing `ranking`, ordered from highest predicted assay fitness
  to lowest.
- Validity: all `Mxx` IDs must occur exactly once. Partial, duplicated, or unknown IDs make the
  output invalid.
- Metric: Spearman correlation between the predicted order and experimental DMS fitness.
- Direction: higher experimental fitness always means a higher value of the measured property.

Do not inspect split scores or previous result files while reasoning about a benchmark case.
Those files are used only by the deterministic scorer after the model has answered.

