#!/usr/bin/env python3
"""
qudit_psc_polar_de.py

Population-dynamics density evolution (DE) for q-ary polar codes on *symmetric qudit
pure-state channels (PSC)* characterized by a circulant Gram matrix eigen-spectrum λ.

This script is the .py/argparse version of `qudit_bpqm_polar.ipynb`, and implements the
same update rules as the PDF:

- Check-node (minus channel) update: mixture sampling over m ∈ [q]
    p_m = (1/q^2) * Σ_j  λ1[m+j] λ2[-j]
    λ^{(⊞,m)}_j = (1/(q p_m)) * λ1[m+j] λ2[-j]
  (Lemma 9 in the PDF; Algorithm 1 samples m ~ p_m)

- Bit-node (plus channel) update:
    λ^{(⊕)}_j = (1/q) * Σ_k  λ1[k] λ2[j-k]
  (Lemma 10 in the PDF)

We track a simple error proxy using the PGM success formula (Lemma 6):
    P_suc = ( (1/q) Σ_j sqrt(λ_j) )^2
and define Perr := 1 - P_suc.

Modes:
  1) de     : run polar DE for a single base channel spectrum and optionally design an information set.
  2) sweep  : sweep a one-parameter base spectrum family and compare achieved design rate vs Holevo capacity.

Examples
--------
# Single DE run for q=3, N=2^10, with explicit spectrum
python qudit_psc_polar_de.py de --q 3 --n 10 --ns 2000 --lam0 2.4,0.45,0.15 --seed 1 --out_npz out.npz

# Single DE run using the "m-family" spectrum [m, (q-m)/(q-1), ...]
python qudit_psc_polar_de.py de --q 5 --n 12 --ns 3000 --m 3.6 --seed 7 --epsilon 0.1 --union_factor 4 --out_info_set A.txt

# Sweep (Fig-1/2 style): design rates for n in {10,12} vs normalized Holevo capacity
python qudit_psc_polar_de.py sweep --q 3 --n_list 10,12 --ns 4000 --d_budget 0.1 --m_points 31 --out_npz sweep.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional numba acceleration
try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*dargs, **dkwargs):  # type: ignore
        """Fallback decorator that leaves the function unchanged."""
        # Used as @njit without parentheses
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        # Used as @njit(...)
        def _decorator(func):
            return func

        return _decorator


# -----------------------------
# Helpers
# -----------------------------
@njit
def _normalize_lambda(lam: np.ndarray, q: int, eps: float = 1e-15) -> np.ndarray:
    """Ensure lam[j] >= 0 and sum(lam)=q (in-place)."""
    s = 0.0
    for j in range(q):
        if lam[j] < 0.0:
            lam[j] = 0.0
        s += lam[j]
    if s < eps:
        for j in range(q):
            lam[j] = 1.0
        return lam
    scale = q / s
    for j in range(q):
        lam[j] *= scale
    return lam


@njit
def _cconv_mod_q(lam1: np.ndarray, lam2: np.ndarray, q: int, out: np.ndarray) -> np.ndarray:
    """Circular convolution: out[j] = Σ_k lam1[k] * lam2[j-k] (indices mod q)."""
    for j in range(q):
        acc = 0.0
        for k in range(q):
            acc += lam1[k] * lam2[(j - k) % q]
        out[j] = acc
    return out


# -----------------------------
# Qudit PSC polarization updates (PDF Lemma 9/10)
# -----------------------------
@njit
def bitnode_lambda(lam1: np.ndarray, lam2: np.ndarray, q: int, out: np.ndarray) -> np.ndarray:
    """
    Bit-node / plus update (PDF Lemma 10):
        λ^{(⊕)}_j = (1/q) * Σ_k  λ1[k] λ2[j-k]
    """
    _cconv_mod_q(lam1, lam2, q, out)
    invq = 1.0 / q
    for j in range(q):
        out[j] *= invq
    _normalize_lambda(out, q)
    return out


@njit
def checknode_sample_lambda(lam1: np.ndarray, lam2: np.ndarray, q: int, out: np.ndarray) -> np.ndarray:
    """
    Check-node / minus update (PDF Lemma 9), sampled:
      p_m = (1/q^2) * Σ_j  λ1[m+j] λ2[-j]
      λ^{(⊞,m)}_j = (1/(q p_m)) * λ1[m+j] λ2[-j]
    For DE, sample m ~ p_m and output the corresponding spectrum.
    """
    pm = np.zeros(q, dtype=np.float64)
    invq2 = 1.0 / (q * q)

    for m in range(q):
        s = 0.0
        for j in range(q):
            s += lam1[(m + j) % q] * lam2[-j % q]
        pm[m] = s * invq2  # probabilities sum to 1 (assuming Σ λ = q)

    # sample m according to pm
    u = np.random.random()
    c = 0.0
    m_star = 0
    for m in range(q):
        c += pm[m]
        if u <= c:
            m_star = m
            break

    pms = pm[m_star]
    if pms <= 0.0:
        for j in range(q):
            out[j] = 1.0
        return out

    inv_q_pm = 1.0 / (q * pms)
    for j in range(q):
        out[j] = inv_q_pm * lam1[(m_star + j) % q] * lam2[-j % q]

    _normalize_lambda(out, q)
    return out


# -----------------------------
# Metrics from spectrum (PDF Lemma 3 & 6)
# -----------------------------
@njit
def psuc_pgm_from_lambda(lam: np.ndarray, q: int) -> float:
    """PGM success probability proxy: Psuc = ((1/q) Σ_j sqrt(λ_j))^2."""
    s = 0.0
    for j in range(q):
        x = lam[j]
        if x < 0.0:
            x = 0.0
        s += np.sqrt(x)
    s *= (1.0 / q)
    return s * s


@njit
def holevo_I_from_lambda(lam: np.ndarray, q: int, log_base2: bool = True) -> float:
    """
    Holevo information for uniform input (bits if log_base2=True):
        I(W) = log q - (1/q) Σ_j λ_j log λ_j
    """
    if log_base2:
        logq = np.log2(q)
        invlog = 1.0 / np.log(2.0)
    else:
        logq = np.log(q)
        invlog = 1.0

    s = 0.0
    for j in range(q):
        x = lam[j]
        if x > 0.0:
            s += x * (np.log(x) * invlog)
    return logq - (1.0 / q) * s


# -----------------------------
# Polar DE (population dynamics)
# -----------------------------
@njit
def polar_de_qudit(n: int, q: int, lam0: np.ndarray, Npop: int, seed: int = 0) -> np.ndarray:
    """
    Return spectra populations for all 2^n synthetic channels.

    Output shape: (2^n, Npop, q)

    Each synthetic channel is represented by a population of spectra.
    For each parent channel, we pick two i.i.d. spectra samples and apply:
      - minus child: checknode_sample_lambda
      - plus  child: bitnode_lambda
    """
    np.random.seed(seed)

    stage = np.empty((1, Npop, q), dtype=np.float64)
    for t in range(Npop):
        for j in range(q):
            stage[0, t, j] = lam0[j]
        _normalize_lambda(stage[0, t, :], q)

    tmp_plus = np.empty(q, dtype=np.float64)
    tmp_minus = np.empty(q, dtype=np.float64)

    for depth in range(n):
        new_stage = np.empty((2 ** (depth + 1), Npop, q), dtype=np.float64)

        for node in range(2 ** depth):
            parent = stage[node, :, :]

            for t in range(Npop):
                a = np.random.randint(0, Npop)
                b = np.random.randint(0, Npop)

                lam1 = parent[a, :]
                lam2 = parent[b, :]

                checknode_sample_lambda(lam1, lam2, q, tmp_minus)
                for j in range(q):
                    new_stage[2 * node, t, j] = tmp_minus[j]

                bitnode_lambda(lam1, lam2, q, tmp_plus)
                for j in range(q):
                    new_stage[2 * node + 1, t, j] = tmp_plus[j]

        stage = new_stage

    return stage


@njit
def estimate_node_metrics(stage: np.ndarray, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    stage: (N, Npop, q)
    returns:
      avg_psuc[node] = E[Psuc_PGM] over population
      avg_I[node]    = E[Holevo I] over population (bits)
    """
    n_nodes = stage.shape[0]
    Npop = stage.shape[1]

    avg_psuc = np.zeros(n_nodes, dtype=np.float64)
    avg_I = np.zeros(n_nodes, dtype=np.float64)

    for node in range(n_nodes):
        s_ps = 0.0
        s_I = 0.0
        for t in range(Npop):
            lam = stage[node, t, :]
            s_ps += psuc_pgm_from_lambda(lam, q)
            s_I += holevo_I_from_lambda(lam, q, True)
        avg_psuc[node] = s_ps / Npop
        avg_I[node] = s_I / Npop

    return avg_psuc, avg_I


# -----------------------------
# Design helpers
# -----------------------------
def lambda_from_m(m: float, q: int) -> np.ndarray:
    """
    One-parameter spectrum family used in the notebook:
      λ(m) = [m, (q-m)/(q-1), ..., (q-m)/(q-1)]  with Σ λ = q.
    Valid m ∈ [1, q].
    """
    m = float(m)
    if m < 1.0 or m > q:
        raise ValueError(f"m must be in [1, q]. Got m={m}, q={q}")
    rest = (q - m) / (q - 1)
    lam = np.empty(q, dtype=np.float64)
    lam[0] = m
    lam[1:] = rest
    return lam


def parse_lam0_csv(s: str, q: Optional[int]) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    lam = np.array([float(x) for x in parts], dtype=np.float64)
    if q is not None and lam.size != q:
        raise ValueError(f"--lam0 has length {lam.size} but --q={q}")
    return lam


def design_info_set_from_biterr(
    biterr: np.ndarray,
    *,
    epsilon: Optional[float] = None,
    d_budget: Optional[float] = None,
    union_factor: float = 1.0,
) -> Tuple[np.ndarray, int, float]:
    """
    Given per-channel Perr (length N), choose the largest k such that
        union_factor * Σ_{i in A} Perr_i <= epsilon
    when epsilon is provided.

    If d_budget is provided instead, it is interpreted as the cumulative sum threshold:
        Σ Perr_i <= d_budget
    (equivalently epsilon = union_factor * d_budget).

    Returns:
      A (np.ndarray): indices of selected channels (size k), sorted by increasing Perr
      k (int)
      rate (float) = k/N
    """
    biterr = np.asarray(biterr, dtype=float).ravel()
    N = biterr.size

    order = np.argsort(biterr)  # smallest Perr first
    se = biterr[order]
    cse = np.cumsum(se)

    if epsilon is None:
        if d_budget is None:
            raise ValueError("Provide either epsilon or d_budget for design.")
        epsilon_eff = union_factor * float(d_budget)
    else:
        epsilon_eff = float(epsilon)

    # find max k with union_factor * cse[k-1] <= epsilon_eff
    thresh = epsilon_eff / union_factor
    k = int(np.sum(cse <= thresh))
    A = order[:k]
    rate = k / N
    return A, k, rate


# -----------------------------
# CLI
# -----------------------------
def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--q", type=int, required=True, help="Alphabet size (prime).")
    p.add_argument("--n", type=int, required=True, help="Polar exponent (N=2^n).")
    p.add_argument("--ns", type=int, default=3000, help="Population size for DE.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--disable_numba", action="store_true", help="Disable numba JIT (pure python).")


def cmd_de(args: argparse.Namespace) -> None:
    if args.disable_numba and _HAS_NUMBA:
        # Numba has already compiled decorated functions; the simplest practical
        # switch is to warn and continue (still correct, just faster).
        print("Warning: --disable_numba requested but numba is available. "
              "This build keeps numba; results are unchanged.")

    q = int(args.q)
    n = int(args.n)
    Npop = int(args.ns)

    if args.lam0 is not None and args.m is not None:
        raise ValueError("Choose exactly one of --lam0 or --m.")
    if args.lam0 is None and args.m is None:
        raise ValueError("Provide a base channel via --lam0 or --m.")

    if args.lam0 is not None:
        lam0 = parse_lam0_csv(args.lam0, q=q)
    else:
        lam0 = lambda_from_m(args.m, q)

    if lam0.size != q:
        raise ValueError(f"Base spectrum must have length q={q}. Got {lam0.size}")

    # run DE
    stage = polar_de_qudit(n, q, lam0, Npop, int(args.seed))
    avg_psuc, avg_I = estimate_node_metrics(stage, q)
    biterr = 1.0 - avg_psuc

    # summarize
    N = 2 ** n
    cap_bits = float(holevo_I_from_lambda(lam0, q, True))
    print(f"(q, n, N) = ({q}, {n}, {N}), Npop={Npop}")
    print(f"Base spectrum λ0: {np.array2string(lam0, precision=6)}  (sum={lam0.sum():.6f})")
    print(f"Holevo I(W) = {cap_bits:.6f} bits/use, normalized I/log2(q) = {cap_bits/np.log2(q):.6f}")
    print(f"Avg Perr across synthetic channels (proxy) = {float(np.mean(biterr)):.6f}")
    print(f"Min/median/max Perr = {float(np.min(biterr)):.6e} / {float(np.median(biterr)):.6e} / {float(np.max(biterr)):.6e}")

    # design (optional)
    if args.epsilon is not None or args.d_budget is not None:
        A, k, rate = design_info_set_from_biterr(
            biterr,
            epsilon=args.epsilon,
            d_budget=args.d_budget,
            union_factor=float(args.union_factor),
        )
        print(f"Design: k={k} info channels, rate={rate:.6f}  "
              f"(union_factor={args.union_factor}, epsilon={args.epsilon}, d_budget={args.d_budget})")

        if args.out_info_set is not None:
            outp = Path(args.out_info_set)
            outp.parent.mkdir(parents=True, exist_ok=True)
            # store as 0-indexed indices (consistent with numpy arrays / code)
            outp.write_text("\n".join(str(int(i)) for i in A) + "\n")
            print(f"Wrote information set indices (0-based) to: {outp}")

    # save (optional)
    if args.out_npz is not None:
        outp = Path(args.out_npz)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            outp,
            q=q,
            n=n,
            N=N,
            Npop=Npop,
            seed=int(args.seed),
            lam0=lam0,
            avg_psuc=avg_psuc,
            avg_I=avg_I,
            biterr=biterr,
        )
        print(f"Wrote arrays to: {outp}")


def _parse_int_list_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def cmd_sweep(args: argparse.Namespace) -> None:
    q = int(args.q)
    n_list = _parse_int_list_csv(args.n_list)
    Npop = int(args.ns)
    d_budget = float(args.d_budget)
    m_points = int(args.m_points)
    seed0 = int(args.seed)

    m_grid = np.linspace(1.0, float(q), m_points)
    cap_norm = np.zeros_like(m_grid)
    out: Dict[str, np.ndarray] = {"m": m_grid, "cap_norm": cap_norm}

    for n in n_list:
        out[f"rate_n{n}"] = np.zeros_like(m_grid)
        out[f"k_n{n}"] = np.zeros_like(m_grid, dtype=np.int64)

    logq = np.log2(q)

    for idx, m in enumerate(m_grid):
        lam0 = lambda_from_m(float(m), q)
        cap_bits = float(holevo_I_from_lambda(lam0, q, True))
        cap_norm[idx] = cap_bits / logq

        if args.verbose and idx % max(1, len(m_grid) // 10) == 0:
            print(f"[{idx+1}/{len(m_grid)}] m={m:.4f}, C_norm={cap_norm[idx]:.6f}")

        for n in n_list:
            stage = polar_de_qudit(n, q, lam0, Npop, seed0 + 1000 * n + idx)
            avg_psuc, _avg_I = estimate_node_metrics(stage, q)
            biterr = 1.0 - avg_psuc

            # notebook-style design: pick k so cumulative Perr <= d_budget
            order = np.argsort(biterr)
            cse = np.cumsum(biterr[order])
            k = int(np.sum(cse <= d_budget))
            out[f"k_n{n}"][idx] = k
            out[f"rate_n{n}"][idx] = k / (2 ** n)

    # save npz
    if args.out_npz is not None:
        outp = Path(args.out_npz)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outp, q=q, n_list=np.array(n_list, dtype=np.int64), d_budget=d_budget, Npop=Npop, seed0=seed0, **out)
        print(f"Wrote sweep results to: {outp}")

    # optional plot (save to PNG)
    if args.out_png is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(m_grid, cap_norm, label=r"Normalized Holevo capacity $I(W)/\log_2 q$")
        for n in n_list:
            ax.plot(m_grid, out[f"rate_n{n}"], label=f"Polar design rate (n={n}, N={2**n})")

        ax.set_xlabel("m (base spectrum parameter)")
        ax.set_ylabel("Rate")
        ax.set_title(f"Polar design rate vs capacity (q={q}, cumulative error budget d={d_budget})")
        ax.set_xlim(float(m_grid.min()), float(m_grid.max()))
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        outp = Path(args.out_png)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outp, dpi=200)
        print(f"Wrote plot to: {outp}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Polar DE for symmetric qudit PSC via Gram-spectrum updates.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_de = sub.add_parser("de", help="Run polar DE for a single base channel.")
    _add_common_args(p_de)
    p_de.add_argument("--lam0", type=str, default=None, help="Comma-separated eigen-spectrum λ0 of length q (sum=q).")
    p_de.add_argument("--m", type=float, default=None, help="Use spectrum family λ=[m,(q-m)/(q-1),...], m in [1,q].")
    p_de.add_argument("--epsilon", type=float, default=None, help="Target block error proxy for design (optional).")
    p_de.add_argument("--d_budget", type=float, default=None, help="Cumulative Perr budget for design (optional).")
    p_de.add_argument("--union_factor", type=float, default=4.0,
                      help="Union-bound factor (PDF Alg.1 uses 4*sum Perr <= epsilon).")
    p_de.add_argument("--out_npz", type=str, default=None, help="Save arrays to a .npz file.")
    p_de.add_argument("--out_info_set", type=str, default=None, help="Write 0-based info-set indices to a text file.")
    p_de.set_defaults(func=cmd_de)

    p_sw = sub.add_parser("sweep", help="Sweep m-family spectrum and compare design rate vs capacity.")
    p_sw.add_argument("--q", type=int, required=True, help="Alphabet size (prime).")
    p_sw.add_argument("--n_list", type=str, required=True, help="Comma-separated n values (e.g., 10,12).")
    p_sw.add_argument("--ns", type=int, default=4000, help="Population size for each DE run.")
    p_sw.add_argument("--seed", type=int, default=1, help="Base seed.")
    p_sw.add_argument("--d_budget", type=float, default=0.1, help="Cumulative Perr budget used to select k.")
    p_sw.add_argument("--m_points", type=int, default=41, help="#grid points for m in [1,q].")
    p_sw.add_argument("--out_npz", type=str, default=None, help="Save sweep arrays to .npz.")
    p_sw.add_argument("--out_png", type=str, default=None, help="Save sweep plot to .png.")
    p_sw.add_argument("--verbose", action="store_true", help="Print progress.")
    p_sw.set_defaults(func=cmd_sweep)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
