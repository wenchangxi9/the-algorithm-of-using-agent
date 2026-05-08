# Raw Community Notes data

The full public Community Notes export is not stored in this repository because it is too large for a normal Git history.

Place the extracted TSV export here when reproducing the matrix-factorization step:

```text
data/raw_communitynotes/extracted/
  notes/
    notes-*.tsv
  noteRatings/
    ratings-*.tsv
  noteStatusHistory/
    noteStatusHistory-00000.tsv
```

The MF clustering script expects this extracted layout. The current packaged artifacts were produced from the 2026-04-07 public export used in the experiments.
