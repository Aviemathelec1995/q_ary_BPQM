# Q-ary PSC BPQM Density Evolution

Simulation code associated with the paper [Belief Propagation with Quantum Messages for Symmetric Q-ary Pure-State Channels](https://arxiv.org/abs/2601.21330).

This repository implements population-dynamics density evolution for belief propagation with quantum messages (BPQM) on symmetric q-ary pure-state channels (PSCs) with circulant Gram matrices. The code tracks channel evolution through Gram-matrix eigenvalue spectra, evaluates PGM symbol-error proxies, estimates regular LDPC decoding thresholds, and constructs polar-code design curves for q-ary PSCs.

## Table of Contents

- [Background](#background)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Examples](#examples)
- [Generated Outputs](#generated-outputs)
- [Notes](#notes)
- [Citation](#citation)
- [License](#license)

## Background

The paper develops BPQM density evolution for symmetric q-ary PSCs whose output Gram matrix is circulant. For this channel class, bit-node and check-node operations can be represented by closed-form recursions on the Gram-matrix eigenvalue list. This removes dependence on a particular state-vector realization and enables classical simulation of BPQM message evolution.

The simulations in this repository focus on:

- Symmetric q-ary PSCs parameterized by eigenvalue spectra summing to `q`.
- Check-node sampling and bit-node convolution updates for BPQM.
- PGM success probability and symmetric Holevo information computed from eigenvalue spectra.
- Population-dynamics density evolution for regular `(dv, dc)` LDPC ensembles.
- Polar density evolution, information-set selection, design-curve generation, and rate-capacity sweeps.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `qudit_psc_ldpc_threshold.py` | LDPC threshold estimator for a regular `(dv, dc)` ensemble on the one-parameter symmetric PSC spectrum family `lambda = [lam, (q-lam)/(q-1), ...]`. It runs BPQM population-dynamics DE, applies a PGM error criterion, and compares the result with the Holevo threshold. |
| `qudit_psc_polar_de.py` | Core polar-code DE implementation. It implements BPQM check-node and bit-node eigenvalue-list recursions, PGM success probability, Holevo information, polar synthetic-channel evolution, information-set design, and sweep utilities. |
| `qudit_psc_polar_plots.py` | Plotting interface for polar-code experiments. It generates sorted synthetic-channel error curves and achieved-rate versus normalized Holevo-capacity curves. |
| `requirements.txt` | Python dependency list for local execution. |

## Requirements

- Python 3.10 or newer.
- Scientific Python packages listed in `requirements.txt`.

Dependencies:

- `numpy`
- `matplotlib`
- `numba`

`numba` is used for JIT acceleration in the polar DE implementation. The polar DE module also contains a pure-Python decorator fallback when `numba` is unavailable.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd q_ary_BPQM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root so local module imports resolve correctly. If commands are launched from another directory, add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/q_ary_BPQM:${PYTHONPATH}"
```

## Examples

Estimate a BPQM-DE threshold for a regular LDPC ensemble:

```bash
python qudit_psc_ldpc_threshold.py \
  --q 5 --dv 3 --dc 6 --ns 3000 --depth 50 --tol 0.05
```

Run polar density evolution for a single q-ary PSC spectrum and write array data:

```bash
python qudit_psc_polar_de.py de \
  --q 3 --n 10 --ns 2000 --lam0 2.4,0.45,0.15 \
  --seed 1 --out_npz polar_de_q3_n10.npz
```

Run polar density evolution using the one-parameter spectrum family and write an information set:

```bash
python qudit_psc_polar_de.py de \
  --q 5 --n 12 --ns 3000 --m 3.6 \
  --seed 7 --epsilon 0.1 --union_factor 4 \
  --out_info_set info_set_q5_n12.txt
```

Generate polar design curves:

```bash
python qudit_psc_polar_plots.py design-curve \
  --q 3 --n_list 7,8,9,10 --lam0 2.4,0.45,0.15 \
  --Npop 2000 --out design_curve.png --no_show
```

Generate achieved-rate versus normalized Holevo-capacity curves:

```bash
python qudit_psc_polar_plots.py rate-vs-capacity \
  --q 3 --n_list 8,10,12 --d_budget 0.1 \
  --Npop 4000 --out rate_vs_cap.png --no_show
```

## Generated Outputs

The scripts can write:

- `.npz` files containing spectra-derived metrics, synthetic-channel error arrays, capacities, rates, and information-set sizes.
- Text files containing zero-indexed polar information-set indices.
- Plot files such as `design_curve.png` and `rate_vs_cap.png`.

Generated numerical arrays and figures are excluded by `.gitignore` unless explicitly added.

## Notes

- The implementation assumes q-ary cyclic indexing modulo `q`.
- The paper states the main q-ary construction for prime `q`; the eigenvalue-list recursions depend on Fourier diagonalization of the circulant Gram matrix.
- All spectra are normalized to satisfy `sum(lambda) = q`.
- The PGM expression is used as the symbol-error proxy for evolved symmetric PSCs.
- LDPC threshold estimation uses finite population size, finite DE depth, and random sampling for check-node herald outcomes.
- Polar-code design selects synthetic channels by cumulative PGM-error budget.

## Citation

If you use this code, cite:

```bibtex
@article{mandal2026bpqm,
  title={Belief Propagation with Quantum Messages for Symmetric Q-ary Pure-State Channels},
  author={Mandal, Avijit and Pfister, Henry D.},
  journal={arXiv preprint arXiv:2601.21330},
  year={2026}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
