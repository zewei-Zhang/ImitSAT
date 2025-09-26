"""
Report median reduced propagation ratio (MRPP = B/A on 'propagations') and Win@1%
comparing ImitSAT (B) to MiniSAT baseline (A) from saved JSON stats.
"""
import os
import glob
import json
import numpy as np

METRICS = ["propagations"]

DELTA = 0.01
EPS = 1e-9


def get_stats_block(item, role):
    """
    Select a stats dict from an item.

    Args:
    	item: One JSON record with stats.
    	role: One of {'raw','key','method'}.

    Returns:
    	Chosen stats dict or empty dict.
    """
    if role == "raw":
        return item.get("raw_stats", {}) or {}
    if role == "key":
        return item.get("key_stats", {}) or {}
    if role == "method":
        # Our method block is imitsat_stats
        return item.get("imitsat_stats", {}) or {}
    return {}


def read_metric(block, metric):
    """
    Read a numeric metric from a stats dict.

    Args:
    	block: Stats dictionary.
    	metric: Metric key, e.g., 'propagations'.

    Returns:
    	Float value or None if missing/invalid.
    """
    if not isinstance(block, dict):
        return None
    v = block.get(metric, None)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def ratio_eps(numer, denom, eps=EPS):
    """
    Safe division numer/denom, replacing denom=0 by eps.

    Args:
    	numer: Numerator.
    	denom: Denominator.
    	eps: Small constant to avoid division by zero.

    Returns:
    	float result of safe division.
    """
    """Compute numer/denom; if denom==0 use eps to avoid division by zero."""
    return numer / (denom if denom != 0.0 else eps)


def analyze_file(items):
    """
    Compute MRPP (median of B/A) and Win@1% per metric for one file.

    Args:
    	items: List of JSON records.

    Returns:
    	Dict metric → summary with mrpp_med, n_ratio, win_ratio, wins, n_win_den.
    """
    out = {}
    for m in METRICS:
        ratios = []        # B/A per instance
        wins = 0           # count of wins at 1%
        n_win_den = 0      # comparisons where A != 0

        for it in items:
            a_blk = get_stats_block(it, "raw")
            b_blk = get_stats_block(it, "method")
            if not a_blk or not b_blk:
                continue

            A = read_metric(a_blk, m)  # baseline (MiniSat)
            B = read_metric(b_blk, m)  # method (ImitSAT)
            if A is None or B is None:
                continue
            try:
                a = float(A); b = float(B)
            except Exception:
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue

            # MRPP uses B/A
            ratios.append(ratio_eps(b, a, EPS))

            # Win@1% (skip when A==0)
            if a != 0.0:
                n_win_den += 1
                if b <= (1.0 - DELTA) * a:
                    wins += 1

        if ratios:
            mrpp_med = float(np.median(np.array(ratios, dtype=float)))
            n_ratio = len(ratios)
        else:
            mrpp_med = None
            n_ratio = 0

        win_ratio = (wins / n_win_den) if n_win_den > 0 else None
        out[m] = {
            "mrpp_med": mrpp_med,
            "n_ratio": n_ratio,
            "win_ratio": win_ratio,
            "wins": wins,
            "n_win_den": n_win_den,
        }
    return out


def print_file_summary(filename, summaries):
    """
    Pretty‑print MRPP and Win@1% for a single file.

    Args:
    	filename: Base file name.
    	summaries: Output from analyze_file().
    """
    print(f"FILE: {filename}")
    for m in METRICS:
        s = summaries[m]
        # MRPP
        if s["n_ratio"] == 0 or s["mrpp_med"] is None:
            mrpp_str = "NA (n=0)"
        else:
            mrpp_str = f"{s['mrpp_med']:.6f} (n={s['n_ratio']})"
        # Win@1%
        if s["n_win_den"] == 0 or s["win_ratio"] is None:
            win_str = "NA (n=0)"
        else:
            win_str = f"{s['win_ratio']:.3f} (n={s['n_win_den']})"

        print(f"  {m}: MRPP={mrpp_str}  Win@1%={win_str}")
    print("")


def main(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        print(f"No .json files in {folder}")
        return

    print("LEGEND:")
    print("  A = raw_stats (MiniSat baseline)")
    print("  B = imitsat_stats (ImitSAT method)")
    print("  MRPP := median of B/A across instances (values < 1 mean improvement).")
    print("  Win@1% := fraction of instances where B <= 0.99 * A (A=0 skipped).")
    print("")

    global_ratios = {m: [] for m in METRICS}   # for MRPP (B/A)
    global_wins = {m: 0 for m in METRICS}
    global_den  = {m: 0 for m in METRICS}

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Skipping {path}: expected a list of dicts")
            continue

        summaries = analyze_file(data)
        print_file_summary(os.path.basename(path), summaries)

        # accumulate globals instance-by-instance for precision
        for it in data:
            for m in METRICS:
                a_blk = get_stats_block(it, "raw")
                b_blk = get_stats_block(it, "method")
                if not a_blk or not b_blk:
                    continue
                A = read_metric(a_blk, m)
                B = read_metric(b_blk, m)
                if A is None or B is None:
                    continue
                try:
                    a = float(A); b = float(B)
                except Exception:
                    continue
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue

                global_ratios[m].append(ratio_eps(b, a, EPS))  # B/A
                if a != 0.0:
                    global_den[m] += 1
                    if b <= (1.0 - DELTA) * a:
                        global_wins[m] += 1

    # FINAL across all files
    print("FINAL (across all files)")
    for m in METRICS:
        ratios = np.array(global_ratios[m], dtype=float)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            print(f"  {m}: MRPP=NA (n=0)  Win@1%=NA (n=0)")
            continue

        mrpp_med = float(np.median(ratios))
        n_ratio = int(ratios.size)

        if global_den[m] == 0:
            win_str = "NA (n=0)"
        else:
            win_ratio = global_wins[m] / global_den[m]
            win_str = f"{win_ratio:.3f} (n={global_den[m]})"

        print(f"  {m}: MRPP={mrpp_med:.6f} (n={n_ratio})  Win@1%={win_str}")
    print("")


if __name__ == "__main__":
    folder = "./output/imitsat"
    main(folder)
