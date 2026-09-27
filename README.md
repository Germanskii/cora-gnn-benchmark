# Cora GNN Benchmark

[![Checks](https://github.com/Germanskii/cora-gnn-benchmark/actions/workflows/checks.yml/badge.svg)](https://github.com/Germanskii/cora-gnn-benchmark/actions/workflows/checks.yml)
Compare GCN, GAT and Graph Transformer for transductive node classification on Cora.

## Experiment

GCN, GAT and Graph Transformer use the standard Cora train/validation/test masks. Each architecture runs with seeds 42, 43 and 44. Validation accuracy controls early stopping; the best state is deep-copied before test evaluation. Outputs include accuracy, macro F1, parameter counts, training time, per-epoch histories, checkpoints and a comparison plot.

This is a transductive benchmark: all graph edges and features are visible during training, while supervised loss uses only training labels. Architectures do not have matched parameter budgets. CUDA scatter operations may be nondeterministic even with fixed seeds.

## Quick start

Python 3.12 was used for local validation. Run commands from the repository root.

```bash
git clone https://github.com/Germanskii/cora-gnn-benchmark.git
cd cora-gnn-benchmark
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.benchmark
```

For the PyTorch projects, the pinned versions reproduce the tested CPU environment. Install compatible GPU wheels for your platform before running on CUDA.

```bash
python -m src.benchmark --epochs 10 --patience 5 --seeds 42
```

[Open in Colab](https://colab.research.google.com/github/Germanskii/cora-gnn-benchmark/blob/main/notebooks/experiment.ipynb). The cleaned Colab/Jupyter experiment is in [`notebooks/experiment.ipynb`](notebooks/experiment.ipynb). To open it locally, install Jupyter separately (`python -m pip install jupyterlab`) and run `jupyter lab`. Command-line runs save figures instead of requiring an interactive window.

## Data

Cora is downloaded by `torch_geometric.datasets.Planetoid` on the first run into `data/Cora`. This repository does not redistribute the dataset. Consult the upstream dataset documentation and applicable terms before reuse.

## Validation and results

Real-data benchmark pending. Synthetic graph execution checks do not establish Cora accuracy. See [VALIDATION.md](VALIDATION.md) for exactly what was checked. No historical notebook output is used as evidence for the corrected implementation.

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

GitHub Actions runs the regression tests and validates notebook structure on pushes and pull requests. It does not download training datasets or establish model accuracy.

## Repository layout

- `src/`: importable implementation and command-line entry points.
- `notebooks/`: cleaned experiment notebook; original explanatory notes are in Russian.
- `tests/`: focused regression checks.
- `requirements.txt`: direct dependency versions used during validation.
- `DATA.md`: data access and redistribution notes.

This project was developed from a university Colab experiment and subsequently cleaned up for reproducibility. Generated data, trained weights and local paths are excluded from version control.


The portfolio cleanup and packaging used AI-assisted development. The notebooks derive from the original university work; validation limits are documented above.
