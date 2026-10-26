Operator Walshness round 3: partial result snapshot

The first blind scan has finished; full research-scope acceptance remains open.
All frozen points listed in tables/points.csv.gz are included in records/*.jsonl.gz.
Records preserve best axes/circuits, every restart W, budgets, seeds, metadata and available posthoc scores.
Use gzip.open(path, "rt") and json.loads(line) to read records; pandas.read_csv reads .csv.gz directly.
Large arrays, full initialization histories and checkpoints stay on the local machine.
Source freeze hashes refer to original unabridged files; compact records cannot reproduce those bytes.
Pure TC/DS, mixed Gibbs and qutrit results are labeled separately.
Levin-Gu frozen transfer to unseen temperatures and 12 spins failed the all-target criterion.
A bounded lambda=0.4 boundary-CZ correction passed independent temperature/size tests through n=12.
It did not pass all nearby-coupling tests; no topological, SPT or phase-string recovery is claimed.
The conditional MLP comparison completed 1728 networks; read supplementary_summary.json and learning tables.
Diagnostic JSON is in diagnostics/part_*.jsonl.gz, including null controls and learning scalar results.
See registration/progress.json for completed coverage and outstanding scope.
The corrected source fixes thermal eigenvalue serialization and uses dense exact t-J flux spectra.
Derived spectrum ordering was checked/repaired after scan completion; NPZ arrays are not exported.

Reproduce from repository root (use the YAML budgets and seeds):
  python scripts/run_signrule_round3.py --help
  python scripts/run_signrule_special.py --help
  python scripts/run_signrule_loop_thermal.py --help
  python scripts/analyze_signrule_round3.py
  python scripts/package_signrule_snapshot.py --out results/NEW_SNAPSHOT
