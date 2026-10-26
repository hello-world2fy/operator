Operator Walshness round 3: partial result snapshot

Experiments remain in progress. This is not final acceptance or a new-physics claim.
All frozen points listed in tables/points.csv.gz are included in records/*.jsonl.gz.
Records preserve best axes/circuits, every restart W, budgets, seeds, metadata and available posthoc scores.
Use gzip.open(path, "rt") and json.loads(line) to read records; pandas.read_csv reads .csv.gz directly.
Large arrays, full initialization histories and checkpoints stay on the local machine.
Source freeze hashes refer to original unabridged files; compact records cannot reproduce those bytes.
Pure TC/DS, mixed Gibbs and qutrit results are labeled separately.
Levin-Gu frozen transfer to unseen temperatures and 12 spins failed the all-target criterion.
No verified physical discovery is claimed; the MLP discovery gate remains closed.
Physical diagnostics include available B/D/E results, not a completed full-grid evaluation.
The historical progress register predates the sparse C Gibbs scan, physics diagnostics and LG transfer.
Remaining work includes complete scoring, larger geometries, selected null controls and final acceptance.
The corrected source fixes thermal eigenvalue serialization and uses dense exact t-J flux spectra.
Thermal NPZ files from early live workers require the documented ordering repair and are not exported.

Reproduce from repository root (use the YAML budgets and seeds):
  python scripts/run_signrule_round3.py --help
  python scripts/run_signrule_special.py --help
  python scripts/run_signrule_loop_thermal.py --help
  python scripts/analyze_signrule_round3.py
  python scripts/package_signrule_snapshot.py --out results/NEW_SNAPSHOT
