# Protocol reference

The original zero-shot implementation in `zeroshot/` is included with local
Python import dependencies, a shared system template, sampling contract and
selected synthetic checks. It retains its protocol version strings and source
hashes. The reference methods did not all use a single universal runner.

## Frozen choices

Single-chain protein/RNA candidates use source-specific cleaning and duplicate
adjudication, stable `(mutation, sequence)` ordering and a fresh NumPy
`RandomState(42)` per assay. The first `min(250, pool_size)` permutation entries
fix candidate identity and sending order. Replacing this with `default_rng`
changes the sample.

Complexes use canonical mutation sites (chain, position, insertion code, WT,
mutant), stable SHA256 identities and seed-42 hash priorities within independent
source/assay scopes. Query sizes depend on the frozen task. Historical reservoir
samples retain their original protocol identity and cannot be relabeled as the
later hash-selected protocol.

ADMET uses all official test rows, rather than N250 mutation sampling. Repeated
SMILES do not automatically collapse distinct official test rows.

Shared zero-shot arms are sequence, sequence plus MSA features, sequence plus
structure features, and sequence plus both. The feature definitions, geometry,
confidence interpretation and missingness guards are part of the protocol.
Unavailable evidence remains unknown. Reference examples and measured query
outcomes are not part of zero-shot input.

Few-shot and multiround implementations are present in the EVAgent reference
package. Their source alone does not provide every final experiment's frozen
support/query split or feedback history. In few-shot runs, only declared support
outcomes are visible. In multiround runs, feedback reveals only candidates already
selected by that method; hidden pool/query outcomes remain evaluator-only.

## Reference checks

```bash
python protocols/zeroshot/check_sampling.py
PYTHONPATH=protocols/zeroshot:baselines/evagent \
  python -m pytest -q protocols/zeroshot/test_*.py
```

The preparation/catalog scripts retain references to separately packaged assets
and historical layouts. Run scientific data preparation only with an explicit
reviewed release manifest and fresh output paths. They are not the portable
`mutabench evaluate` interface.

