# RNA fitness prediction skill card (v1)

This card is a general reasoning aid for zero-shot RNA fitness prediction. It
contains no RNAGym candidate labels, experimental fitness values, rankings, or
assay-specific conclusions. Treat fitness as the phenotype measured by the
assay, not as a synonym for thermodynamic stability.

## 1. Identify the causal layer first

- Restate what higher fitness means in the assay: cleavage, binding,
  aminoacylation, expression, growth, protein stability, or another phenotype.
- For functional RNAs, prioritize preservation of the active fold, catalytic or
  ligand-binding motifs, required flexibility, and long-range pairing.
- For mRNA-coding assays, translate each nucleotide change and classify it as
  synonymous, missense, nonsense, start/stop-altering, or multi-codon. Protein
  structure and function may dominate, while RNA abundance, translation, and
  local RNA structure remain possible secondary mechanisms.
- Do not assume that every mutation is harmful or that a more stable RNA is
  always fitter. The optimum can require conformational switching or local
  accessibility.

## 2. Audit every mutant

- Verify mutation count, positions, wild-type bases, and mutant bases against
  the supplied sequences.
- Separate effects likely to be additive from compensatory combinations. A
  double mutant that restores a base pair can outperform either single change.
- Note overlap with conserved motifs, stems, junctions, loops, termini,
  cleavage sites, ligand pockets, and protein-contact regions.
- Use mutation burden only as a weak prior after mechanistic evidence; it is not
  a score by itself.

## 3. Use structural evidence in a hierarchy

1. Assay-matched experimental structure or chemical-probing evidence.
2. Family covariance and consensus structure (for example, Rfam).
3. Ensemble secondary-structure predictions and base-pair probabilities.
4. A single minimum-free-energy structure.

Record whether evidence is experimental, homologous, or predicted. Prefer
ensemble changes, pairing probabilities, and accessibility over an MFE value
alone. Check construct/chain coverage before transferring annotations from a
PDB structure. Missing evidence must remain missing rather than being invented.

## 4. Use literature without leaking the answer

- General mechanistic papers and method references may supply priors.
- In a zero-shot benchmark, exclude the exact source paper for the evaluated
  assay and any document containing its variant measurements.
- Keep retrieved records frozen and versioned. Record DOI/PMID/URL, retrieval
  date, query, and content hash.
- Distinguish a literature-supported mechanism from a prediction about a
  particular candidate.

## 5. Produce a robust ordering

- Build a compact evidence ledger per mutant: direct motif/structure effect,
  compensatory evidence, coding consequence, weaker priors, and uncertainty.
- Compare candidates under the same rubric. Reserve extreme ranks for strong,
  assay-relevant evidence.
- Before returning JSON, verify exact ID coverage, uniqueness, count, and output
  syntax. Structural or literature evidence never relaxes output validation.

## Evidence limitations

- Experimental structures may describe a homolog, ligand state, truncated
  construct, or only part of the assayed sequence.
- Predicted folds can be unreliable for pseudoknots, ligand-dependent folds,
  protein-bound RNAs, and long RNAs.
- Deep mutational scanning measurements may combine molecular function with
  expression, folding, selection, and host-cell effects.

## General references

- RNAGym benchmark: doi:10.1101/2025.06.16.660049
- ViennaRNA Package 2.0: doi:10.1186/1748-7188-6-26
- Deep mutational scanning overview: doi:10.1038/nmeth.3027
- DMS-MaPseq structural probing: doi:10.1038/nmeth.4057
- Four-base DMS probing: doi:10.1093/nar/gkad522

