"""
qudit_psc_ldpc_threshold.py

Compute the BPQM/PGM-DE threshold (population dynamics) for a q-ary symmetric PSC family
with eigen-spectrum

    lambda0 = [lam, (q-lam)/(q-1), ..., (q-lam)/(q-1)]   (sum = q)

for a regular (dv, dc) LDPC ensemble, and compare to the Holevo (capacity) threshold.

This mirrors the structure of:
https://github.com/Aviemathelec1995/PMBPQM_BSCQ/blob/main/bscq_ldpc_threshold.py
but replaces (delta,gamma) with q-dimensional spectra and uses the qudit PSC
bitnode/checknode rules from the attached PDF.

Usage example:
  python qudit_psc_ldpc_threshold.py --q 8 --dv 3 --dc 6 --ns 3000 --depth 80 --tol 0.05
"""

import numpy as np
import argparse as ap


# -------------------------
# Channel family: lambda-spectrum
# -------------------------
def lambda_family(lam: float, q: int) -> np.ndarray:
    """
    lam in [1, q]. Spectrum sums to q.
      lambda0[0] = lam
      lambda0[1:] = (q - lam)/(q-1)
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    lam = float(lam)
    lam_min = (q - lam) / (q - 1)
    out = np.full(q, lam_min, dtype=float)
    out[0] = lam
    # numerical renorm
    out *= (q / max(out.sum(), 1e-300))
    return out


# -------------------------
# Information / Holevo for symmetric pure-state ensemble
# -------------------------
def holevo_bits_from_lambda(lam_vec: np.ndarray) -> float:
    """
    For uniform ensemble of pure states with Gram eigenvalues lam_vec (sum=q),
    average state eigenvalues are lam_vec/q, so Holevo = S(rho_bar) in bits:
      chi = - sum_j (lam_j/q) log2(lam_j/q)
    """
    q = lam_vec.shape[0]
    p = np.clip(lam_vec / q, 1e-300, 1.0)
    return float(-np.sum(p * np.log2(p)))


# -------------------------
# PGM success proxy from spectrum (Lemma 6):
#   Psuc = ( (1/q) sum_j sqrt(lam_j) )^2
# -------------------------
def pgm_success(lam_vec: np.ndarray) -> float:
    q = lam_vec.shape[0]
    return float((np.sum(np.sqrt(np.clip(lam_vec, 0.0, None))) / q) ** 2)


# -------------------------
# Bitnode / Checknode updates on spectra
# -------------------------
def circ_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution c[m] = sum_k a[m-k] b[k] (indices mod q), O(q^2)."""
    q = a.shape[0]
    c = np.zeros(q, dtype=float)
    for m in range(q):
        s = 0.0
        for k in range(q):
            s += a[(m - k) % q] * b[k]
        c[m] = s
    return c


def bitnode_lambda(lam1: np.ndarray, lam2: np.ndarray) -> np.ndarray:
    """
    Bitnode (Lemma 10): lam_plus[j] = (1/q) * sum_k lam1[k] lam2[j-k]
    That is (1/q) times a circular convolution with the 'j-k' convention.
    Our circ_conv implements sum_k lam1[j-k] lam2[k], so swap args:
      sum_k lam1[k] lam2[j-k] = circ_conv(lam2, lam1)[j]
    """
    q = lam1.shape[0]
    out = circ_conv(lam2, lam1) / q
    # safety renorm: sum=q
    out *= (q / max(out.sum(), 1e-300))
    return out


def checknode_sample_lambda(lam1: np.ndarray, lam2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Checknode (Lemma 9):
      s_m = sum_j lam1[m+j] lam2[-j]  (mod q)
      p_m = s_m / q^2  (and sum_m s_m = q^2 if both sums are q)

      conditional:
        lam^{(m)}_j = (1/(q p_m)) lam1[m+j] lam2[-j]
                   = q * lam1[m+j] lam2[-j] / s_m

    We sample m ~ p and return lam^{(m)}.
    """
    q = lam1.shape[0]
    # compute s_m efficiently using convolution identity:
    # s_m = sum_k lam1[m-k] lam2[k] = circ_conv(lam1, lam2)[m]
    s = circ_conv(lam1, lam2)  # length q, sums to q^2 (ideally)

    total = float(np.sum(s))
    if total <= 1e-300:
        # degenerate fallback
        m = int(rng.integers(0, q))
        s_m = 1e-300
    else:
        # sample categorical with probs s/total
        u = float(rng.random())
        cdf = 0.0
        m = q - 1
        for i in range(q):
            cdf += float(s[i]) / total
            if u < cdf:
                m = i
                break
        s_m = float(s[m])

    # build lam2_neg[j] = lam2[-j]
    idx_neg = (-np.arange(q)) % q
    lam2_neg = lam2[idx_neg]
    # lam1_shift[j] = lam1[m+j] = roll(lam1, -m)[j]
    lam1_shift = np.roll(lam1, -m)

    out = (q * lam1_shift * lam2_neg) / max(s_m, 1e-300)
    # safety renorm: sum=q
    out *= (q / max(out.sum(), 1e-300))
    return out


# -------------------------
# Population dynamics DE for regular LDPC
# -------------------------
def bitnode_vec(pop1: np.ndarray, pop2: np.ndarray, perm: np.ndarray) -> np.ndarray:
    N, q = pop1.shape
    out = np.zeros((N, q), dtype=float)
    for i in range(N):
        out[i] = bitnode_lambda(pop1[i], pop2[perm[i]])
    return out


def checknode_vec(pop1: np.ndarray, pop2: np.ndarray, perm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N, q = pop1.shape
    out = np.zeros((N, q), dtype=float)
    for i in range(N):
        out[i] = checknode_sample_lambda(pop1[i], pop2[perm[i]], rng)
    return out


def bitnode_power(pop: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Combine k incoming messages at a variable node by sequential bitnode folding.
    """
    if k <= 1:
        return pop
    acc = pop.copy()
    base = pop.copy()
    for _ in range(k - 1):
        perm = rng.permutation(pop.shape[0])
        acc = bitnode_vec(acc, base, perm)
    return acc


def checknode_power(pop: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Combine k incoming messages at a check node by sequential checknode folding (with sampling).
    """
    if k <= 1:
        return pop
    acc = pop.copy()
    base = pop.copy()
    for _ in range(k - 1):
        perm = rng.permutation(pop.shape[0])
        acc = checknode_vec(acc, base, perm, rng)
    return acc


def de_run_error_curve(lam0: np.ndarray, N: int, dv: int, dc: int, depth: int, seed: int = 0) -> np.ndarray:
    """
    LDPC recursion mirroring the repo:
      check combine (dc-1)
      bit combine (dv-1)
      then combine with channel message via bitnode
    Returns Pe[t] for t=1..depth-1, using PGM proxy.
    """
    rng = np.random.default_rng(seed)
    q = lam0.shape[0]

    pop = np.repeat(lam0[None, :], N, axis=0)  # depth 0 messages
    chan_pop = pop.copy()

    Pe = np.zeros(depth - 1, dtype=float)

    for t in range(depth - 1):
        # check update: dc-1 incoming
        chk = checknode_power(pop, dc - 1, rng)
        # variable update: dv-1 incoming from checks
        var_extr = bitnode_power(chk, dv - 1, rng)
        # combine with channel
        perm = rng.permutation(N)
        pop = bitnode_vec(chan_pop, var_extr, perm)

        # error proxy at this depth
        Ps = np.array([pgm_success(pop[i]) for i in range(N)], dtype=float)
        Pe[t] = 1.0 - float(np.mean(Ps))

    return Pe


def de_succeeds(lam: float, q: int, N: int, dv: int, dc: int, depth: int,
                eps: float, tail: int, trials: int, seed0: int) -> bool:
    """
    Decide if DE converges (success) for a given lam.

    Criterion:
      success if min(Pe[-tail:]) < eps in a majority of trials.
    """
    lam0 = lambda_family(lam, q)
    wins = 0
    for r in range(trials):
        Pe = de_run_error_curve(lam0, N=N, dv=dv, dc=dc, depth=depth, seed=seed0 + r)
        if float(np.min(Pe[-tail:])) < eps:
            wins += 1
    return wins >= (trials // 2 + 1)


# -------------------------
# Binary searches for thresholds
# -------------------------
def binary_search_de_lambda_max(q: int, N: int, dv: int, dc: int, depth: int,
                                lam_lo: float, lam_hi: float, tol: float,
                                eps: float, tail: int, trials: int, seed0: int) -> float:
    """
    Find the largest lambda such that DE succeeds.
    Assumes monotone: succeeds for smaller lambda, fails for larger lambda.
    """

    lo, hi = lam_lo, lam_hi

    ok_lo = de_succeeds(lo, q, N, dv, dc, depth, eps, tail, trials, seed0)
    ok_hi = de_succeeds(hi, q, N, dv, dc, depth, eps, tail, trials, seed0)

    if not ok_lo:
        raise RuntimeError(
            "DE does not succeed even at lam_lo. Try smaller lam_lo (>=1), increase depth/ns, "
            "or relax eps/tail."
        )
    if ok_hi:
        # It might happen if depth is huge and criterion is too loose; still return hi.
        return hi

    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        if de_succeeds(mid, q, N, dv, dc, depth, eps, tail, trials, seed0):
            lo = mid   # can tolerate more noise → push lambda up
        else:
            hi = mid   # too noisy → lambda too large
    return lo

def binary_search_holevo_lambda_max(q: int, dv: int, dc: int, tol: float) -> float:
    """
    Largest lambda such that Holevo(lambda) >= R*log2(q),
    where R = 1 - dv/dc.
    """
    R = 1.0 - (dv / dc)
    target = R * np.log2(q)

    lo, hi = 1.0, float(q)

    chi_lo = holevo_bits_from_lambda(lambda_family(lo, q))  # ~ log2(q)
    chi_hi = holevo_bits_from_lambda(lambda_family(hi, q))  # ~ 0

    if chi_lo < target:
        return lo
    if chi_hi >= target:
        return hi

    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        chi_mid = holevo_bits_from_lambda(lambda_family(mid, q))
        if chi_mid >= target:
            lo = mid  # still enough info → allow larger lambda
        else:
            hi = mid
    return lo



# -------------------------
# Main
# -------------------------
def main():
    parser = ap.ArgumentParser("Qudit PSC LDPC threshold (BPQM-DE) vs Holevo threshold")
    parser.add_argument("--q", type=int, default=4, help="Alphabet size q")
    parser.add_argument("--dv", type=int, default=3, help="Variable-node degree")
    parser.add_argument("--dc", type=int, default=6, help="Check-node degree")
    parser.add_argument("--ns", type=int, default=2000, help="Population size N (DE samples)")
    parser.add_argument("--depth", type=int, default=80, help="DE iterations / tree depth")
    parser.add_argument("--tol", type=float, default=0.05, help="Binary search tolerance in lambda")

    # DE decision parameters (mirrors repo logic with a small error floor)
    parser.add_argument("--eps", type=float, default=1e-4, help="Success if min tail error < eps")
    parser.add_argument("--tail", type=int, default=10, help="Tail window size for success test")
    parser.add_argument("--trials", type=int, default=3, help="Independent DE trials per lambda (majority vote)")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")

    # search range for lambda
    parser.add_argument("--lam_lo", type=float, default=1.0, help="Lower bracket for lambda search (>=1)")
    parser.add_argument("--lam_hi", type=float, default=None, help="Upper bracket for lambda search (<=q). Default=q.")

    parser.add_argument("--show_curves", action="store_true",
                        help="Print DE tail errors near the found thresholds (no plotting).")

    args = parser.parse_args()

    q, dv, dc = args.q, args.dv, args.dc
    N, depth = args.ns, args.depth
    tol = args.tol
    lam_lo = float(args.lam_lo)
    lam_hi = float(args.lam_hi) if args.lam_hi is not None else float(q)
    for test_lam in [1.0, float(q)]:
        lam0 = lambda_family(test_lam, q)
        Pe = de_run_error_curve(lam0, N=N, dv=dv, dc=dc, depth=depth, seed=args.seed)
    print(f"lambda={test_lam:.3f}: tail min Pe = {np.min(Pe[-args.tail:]):.3e}, final Pe={Pe[-1]:.3e}, Holevo={holevo_bits_from_lambda(lam0):.6f}")


    if not (1.0 <= lam_lo <= lam_hi <= float(q) + 1e-9):
        raise ValueError("Require 1 <= lam_lo <= lam_hi <= q.")

    R = 1.0 - dv / dc
    print(f"(q, dv, dc)=({q}, {dv}, {dc}), design rate R ≈ {R:.6f} symbols/use, target bits/use = {R*np.log2(q):.6f}")

    print("\nSearching BPQM/PGM-DE threshold in lambda (largest lambda that still succeeds) ...")
    lam_de = binary_search_de_lambda_max(
    q=q, N=N, dv=dv, dc=dc, depth=depth,
    lam_lo=lam_lo, lam_hi=lam_hi, tol=tol,
    eps=args.eps, tail=args.tail, trials=args.trials, seed0=args.seed)
    print(f"DE threshold lambda* ≈ {lam_de:.6f}")
    # chi_de = holevo_bits_from_lambda(lambda_family(lam_de, q))
    # print(f"  (Info from DE≈ {chi_de:.6f} bits/use)")
    print("\nComputing Holevo threshold in lambda (largest lambda with chi >= target rate) ...")
    lam_h = binary_search_holevo_lambda_max(q=q, dv=dv, dc=dc, tol=tol)
    chi_at = holevo_bits_from_lambda(lambda_family(lam_h, q))
    print(f"Holevo threshold lambda_H ≈ {lam_h:.6f}  (Holevo ≈ {chi_at:.6f} bits/use)")


    # Optional diagnostics: show tail errors near thresholds
    if args.show_curves:
        for name, lam in [("lambda* (DE)", lam_de), ("lambda_H (Holevo)", lam_h)]:
            lam0 = lambda_family(lam, q)
            Pe = de_run_error_curve(lam0, N=N, dv=dv, dc=dc, depth=depth, seed=args.seed)
            print(f"\n{name}: {lam:.6f}")
            print(f"  tail min(Pe[-{args.tail}:]) = {np.min(Pe[-args.tail:]):.3e}")
            print(f"  final Pe[-1]               = {Pe[-1]:.3e}")
            print(f"  first 10 Pe                = {[float(x) for x in Pe[:10]]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
