Operator Walshness round 3: completed staged experiment, corrected compact results

Read report.html for scientific conclusions, actual coverage and conditional size/gate limits.
3798 primary points; 54 triangular pure scans plus 54 alpha0-start checks; 160 random controls.
1728 learning training instances represent 768 distinct representation/architecture/seed conditions.
All 1512 qutrit optimizations were repeated with float64 kernel constants; older snapshots are superseded for these W values.
Records retain best axes/circuits, all restart W values, metadata, budgets, scalar baselines, feasible degree spectra and posthoc scores.
Use gzip.open(path, "rt") plus json.loads(line) for records/*.jsonl.gz and diagnostics/*.jsonl.gz.
pandas.read_csv reads tables/*.csv.gz directly.
Large density arrays, full reliability masks, initialization histories, checkpoints and archived legacy results remain local.
Freeze hashes refer to original unabridged candidate files; no full raw-result digest sweep was run.
The boundary-CZ correction at lambda0.4 has bounded phase benefits; there is no claimed topological/SPT/path-rule recovery or universal learning advantage.
Reproduce with the YAML budgets and scripts; begin with python -m pytest and inspect registration/requirements_audit.json.

acceptance_progress.json is the historical first-scan repair record. Current acceptance is registration/requirements_audit.json plus registration/report_review.json.
