# Validation

Validated on 2026-09-27 with Python 3.12 on Linux CPU.

Real-data benchmark pending. Synthetic graph execution checks do not establish Cora accuracy.

The source notebook passed schema checks and sequential execution during cleanup. For ML notebooks, external data were replaced only inside the test harness by small explicitly synthetic fixtures, and training was shortened. These checks verify code paths, not scientific performance. Published notebooks retain the full experiment settings and contain no synthetic scores.

The extracted command-line modules are additionally checked for importability, help output and the regression cases included in `tests/`. No training or download occurs on module import.
