# Validation

## Isolated recheck: 29 October 2026

A fresh per-project virtual environment was created on Windows with Python
3.12.10. The pinned runtime and development dependencies installed from wheels;
`pip check` passed. The environment's pip was updated to 26.2.2. The final
`pip-audit` scan reported no known vulnerabilities in that installed package
set at the time of the check. This is a dated advisory check, not a security guarantee.

- The regression test passed, exercising forward output and independent copied
  state for all three architectures.
- The default CLI downloaded real Cora data and completed all nine model/seed
  runs: seeds 42/43/44, maximum 200 epochs, patience 30, standard public masks.
- Notebook schema and Python syntax passed. Every notebook code cell also ran
  sequentially with real data and its original settings, using a Python harness
  with a display helper; this was not an interactive Jupyter session.
- CLI help and focused Ruff syntax/import checks passed. Source/test formatting
  was cleaned up without changing the experiment protocol.
- Aggregate results and raw-data fingerprints are in [docs/results.json](docs/results.json).

CPU execution used two OMP/MKL threads. Timings were collected while other
experiments ran on the same host and are not a controlled hardware comparison.
No GPU run, alternate split, matched parameter budget or broad hyperparameter
search was performed. The prepared changes have not yet run on remote CI.

## Earlier preparation record: 27 September 2026

Validated on 2026-09-27 with Python 3.12 on Linux CPU.

Real-data benchmark pending. Synthetic graph execution checks do not establish Cora accuracy.

The source notebook passed schema checks and sequential execution during cleanup. For ML notebooks, external data were replaced only inside the test harness by small explicitly synthetic fixtures, and training was shortened. These checks verify code paths, not scientific performance. Published notebooks retain the full experiment settings and contain no synthetic scores.

The extracted command-line modules are additionally checked for importability, help output and the regression cases included in `tests/`. No training or download occurs on module import.
