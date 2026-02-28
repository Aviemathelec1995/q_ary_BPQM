#!/usr/bin/env python3
"""
qudit_psc_polar_plots.py

Plot helpers for qudit PSC polar density evolution (DE), matching the plots
used in qudit_bpqm_polar.ipynb:

  (A) "Design curve" / polarization plot:
        x = normalized rank of synthetic channels after sorting by error
        y = sorted channel error (1 - E[P_suc(PGM)])

  (B) "Rate vs capacity" sweep over the base-spectrum family
        lam(m) = [m, (q-m)/(q-1), ..., (q-m)/(q-1)]
      plotting normalized Holevo capacity alongside achieved polar design rate
      for one or more blocklength exponents n (N=2^n).

This file depends on:
  - qudit_psc_polar_de.py (same directory)

Example:
  python qudit_psc_polar_plots.py design-curve --q 3 --n_list 7,8,9,10 --lam0 2.4,0.45,0.15 --Npop 2000 --out design_curve.png
  python qudit_psc_polar_plots.py rate-vs-capacity --q 3 --n_list 8,10,12 --d_budget 0.1 --Npop 4000 --out rate_vs_cap.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import qudit_psc_polar_de as polar


def run_de_and_get_biterr(*, n: int, q: int, lam0: np.ndarray, Npop: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs population DE and returns:
      biterr[i]   = 1 - avg_psuc[i]
      avg_psuc[i] = E[P_suc(PGM)] for synthetic channel i
      avg_I[i]    = E[I(W)] for synthetic channel i
    """
    stage = polar.polar_de_qudit(n, q, lam0, Npop, seed)   # positional args for numba compatibility
    avg_psuc, avg_I = polar.estimate_node_metrics(stage, q)
    biterr = 1.0 - avg_psuc
    return biterr, avg_psuc, avg_I


def plot_polar_design_curves(
    biterr_by_n: Dict[int, np.ndarray],
    *,
    ylim: Tuple[float, float] = (0.0, 1.0),
    title: Optional[str] = None,
    out: Optional[str] = None,
    show: bool = True,
):
    """
    Matches the notebook "Fig-4 style" design curve.

    For each n, sort biterr increasingly and plot:
        x = (1..N)/N, y = sorted(biterr)

    Args:
      biterr_by_n: dict {n: biterr_array_of_len_2^n}
      ylim: y-axis limits
      title: plot title
      out: if provided, save figure to this path
      show: if True, plt.show()
    """
    fig, ax = plt.subplots()

    for n in sorted(biterr_by_n.keys()):
        biterr = np.asarray(biterr_by_n[n], dtype=float).ravel()
        N = 2 ** int(n)
        if biterr.size != N:
            raise ValueError(f"n={n}: expected length {N}, got {biterr.size}")

        y = np.sort(biterr)                # increasing error
        x = (np.arange(N) + 1) / N         # normalized rank in (0,1]
        ax.plot(x, y, label=f"n={n}")
    

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"Normalized Channel Rank $\frac{i}{N}$", fontsize=20)
    ax.set_ylabel("Channel Error Rate ", fontsize=20)
    if title is not None:
        ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=17)
    

    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return ax


def sweep_m_design_rates(
    *,
    q: int,
    n_list: Iterable[int],
    d_budget: float,
    m_grid: np.ndarray,
    Npop: int,
    seed0: int,
    union_factor: float = 4.0,
    verbose: bool = False,
):
    """
    Sweep m over the base-spectrum family and compute:
      cap_norm[m] = Holevo capacity / log2(q)
      rate_n[m]   = k/N where k chosen by cumulative Perr budget (scaled by union_factor)

    Returns a dict suitable for plot_rates_vs_capacity.
    """
    n_list = tuple(int(n) for n in n_list)
    m_grid = np.asarray(m_grid, dtype=float)

    out = {
        "m": m_grid,
        "cap_norm": np.zeros_like(m_grid),
        "q": int(q),
        "d_budget": float(d_budget),
        "union_factor": float(union_factor),
        "n_list": n_list,
    }
    for n in n_list:
        out[f"rate_n{n}"] = np.zeros_like(m_grid)
        out[f"k_n{n}"] = np.zeros_like(m_grid, dtype=int)

    logq = math.log2(q)

    for idx, m in enumerate(m_grid):
        lam0 = polar.lambda_from_m(m, q)

        # normalized Holevo capacity of the base channel
        cap_bits = polar.holevo_I_from_lambda(lam0, q, log_base2=True)
        out["cap_norm"][idx] = cap_bits / logq

        if verbose and (idx % max(1, len(m_grid) // 10) == 0):
            print(f"[{idx+1}/{len(m_grid)}] m={m:.4f}, C_norm={out['cap_norm'][idx]:.4f}")

        for n in n_list:
            biterr, _, _ = run_de_and_get_biterr(n=n, q=q, lam0=lam0, Npop=Npop, seed=seed0 + 997 * idx + 17 * n)
            # Notebook-style: pick k by cumulative sum of sorted errors <= d_budget/union_factor
            # polar.design_info_set_from_biterr returns (A, k, rate)
            _A, k, _rate = polar.design_info_set_from_biterr(
                biterr,
                d_budget=d_budget,
                union_factor=union_factor,
            )
            out[f"k_n{n}"][idx] = int(k)
            out[f"rate_n{n}"][idx] = float(k) / (2 ** n)

    return out


def plot_rates_vs_capacity(
    results: dict,
    *,
    out: Optional[str] = None,
    show: bool = True,
):
    """
    Matches the notebook plot: normalized capacity curve + achieved design rate curves.
    """
    m = np.asarray(results["m"], dtype=float)
    cap = np.asarray(results["cap_norm"], dtype=float)
    q = int(results["q"])
    d_budget = float(results["d_budget"])
    union_factor = float(results.get("union_factor", 4.0))
    n_list = tuple(int(n) for n in results["n_list"])

    fig, ax = plt.subplots()

    ax.plot(m, cap, label=r"$I(W)/\log_2 q$")

    for n in n_list:
        ax.plot(m, results[f"rate_n{n}"], label=f"n={n}")
    

    ax.set_xlabel(r"$\lambda_0$",fontsize=20)
    ax.set_ylabel("Rate",fontsize=20)
    ax.set_title(f"Polar design rate (q={q}, BLER={d_budget})",fontsize=18)
    ax.set_xlim(m.min(), m.max())
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=17)
    # ax.legend()

    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return ax


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    sub = p.add_subparsers(dest="cmd", required=True)

    # -----------------------
    # design-curve
    # -----------------------
    p_dc = sub.add_parser("design-curve", help="Plot sorted channel error vs normalized rank for one or more n.")
    p_dc.add_argument("--q", type=int, required=True, help="Alphabet size q.")
    p_dc.add_argument("--n_list", type=str, required=True, help="Comma-separated n values (N=2^n). Example: 7,8,9,10")
    p_dc.add_argument("--Npop", type=int, default=2000, help="Population size for DE.")
    p_dc.add_argument("--seed0", type=int, default=1, help="Base RNG seed.")
    p_dc.add_argument("--lam0", type=str, default=None, help="Comma-separated base spectrum λ0 (length q, sums to q).")
    p_dc.add_argument("--m", type=float, default=None, help="If set, use base family lam(m) instead of --lam0.")
    p_dc.add_argument("--ylim", type=str, default="0.0,1.0", help="y-axis limits as 'ymin,ymax'.")
    p_dc.add_argument("--title", type=str, default=None, help="Plot title.")
    p_dc.add_argument("--out", type=str, default=None, help="Save plot to file (png/pdf/etc).")
    p_dc.add_argument("--no_show", action="store_true", help="Do not show interactive window.")

    # -----------------------
    # rate-vs-capacity
    # -----------------------
    p_rc = sub.add_parser("rate-vs-capacity", help="Sweep m and plot design rate vs normalized Holevo capacity.")
    p_rc.add_argument("--q", type=int, required=True, help="Alphabet size q.")
    p_rc.add_argument("--n_list", type=str, required=True, help="Comma-separated n values. Example: 8,10,12")
    p_rc.add_argument("--d_budget", type=float, required=True, help="Cumulative error budget d (not multiplied by union_factor).")
    p_rc.add_argument("--union_factor", type=float, default=4.0, help="Union-bound factor (PDF uses 4).")
    p_rc.add_argument("--Npop", type=int, default=4000, help="Population size for DE.")
    p_rc.add_argument("--seed0", type=int, default=1, help="Base RNG seed.")
    p_rc.add_argument("--m_grid", type=str, default=None, help="Comma-separated m values, or omit to use linspace in [1,q].")
    p_rc.add_argument("--m_points", type=int, default=31, help="If --m_grid omitted, number of points in linspace([1,q]).")
    p_rc.add_argument("--out", type=str, default=None, help="Save plot to file (png/pdf/etc).")
    p_rc.add_argument("--no_show", action="store_true", help="Do not show interactive window.")
    p_rc.add_argument("--verbose", action="store_true", help="Print progress during sweep.")

    return p


def cmd_design_curve(args: argparse.Namespace) -> None:
    q = int(args.q)
    n_list = _parse_csv_ints(args.n_list)
    ylim_vals = _parse_csv_floats(args.ylim)
    if len(ylim_vals) != 2:
        raise ValueError("--ylim must be 'ymin,ymax'")

    if (args.lam0 is None) == (args.m is None):
        raise ValueError("Provide exactly one of --lam0 or --m.")

    if args.lam0 is not None:
        lam0 = polar.parse_lam0_csv(args.lam0, q=q)
    else:
        lam0 = polar.lambda_from_m(float(args.m), q=q)

    biterr_by_n: Dict[int, np.ndarray] = {}
    for n in n_list:
        biterr, _, _ = run_de_and_get_biterr(n=n, q=q, lam0=lam0, Npop=int(args.Npop), seed=int(args.seed0) + 31 * n)
        biterr_by_n[int(n)] = biterr

    title = args.title
    if title is None:
        title = f"Polarization channel error curve (q={q})"

    plot_polar_design_curves(
        biterr_by_n,
        ylim=(float(ylim_vals[0]), float(ylim_vals[1])),
        title=title,
        out=args.out,
        show=not args.no_show,
    )


def cmd_rate_vs_capacity(args: argparse.Namespace) -> None:
    q = int(args.q)
    n_list = _parse_csv_ints(args.n_list)

    if args.m_grid is None:
        m_grid = np.linspace(1.0, float(q), int(args.m_points))
    else:
        m_grid = np.asarray(_parse_csv_floats(args.m_grid), dtype=float)

    results = sweep_m_design_rates(
        q=q,
        n_list=n_list,
        d_budget=float(args.d_budget),
        m_grid=m_grid,
        Npop=int(args.Npop),
        seed0=int(args.seed0),
        union_factor=float(args.union_factor),
        verbose=bool(args.verbose),
    )

    plot_rates_vs_capacity(results, out=args.out, show=not args.no_show)


def main() -> None:
    p = build_argparser()
    args = p.parse_args()

    if args.cmd == "design-curve":
        cmd_design_curve(args)
    elif args.cmd == "rate-vs-capacity":
        cmd_rate_vs_capacity(args)
    else:
        raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
