# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
Run CaDiCaL (+ImitSAT guidance) over CNF instances from a TXT list or folders of .cnf files,
writing per‑instance JSON. The ImitSAT model is loaded ONCE per TXT file or per (sub)folder.

Dependencies:
  - integration.ImitSAT_CaDiCaL: ImitSATRunner, ImitSATPropagator (your base integration)
  - utils.utils: read_sat_problems_lines, write_temp_cnf_file, cnf_line_2_CNF_class, save_dicts_to_json
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
from pysat.formula import CNF
from pysat.solvers import Cadical195

from utils.utils import (
    read_sat_problems_lines,
    write_temp_cnf_file,
    cnf_line_2_CNF_class,
    save_dicts_to_json,
)

from integration.cadical.ImitSAT_CaDiCaL import (
    ImitSATRunner,
    ImitSATPropagator,
)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def _int_stat(d: dict, key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default))
    except Exception:
        return default


# --------------- baseline (raw CaDiCaL) ---------------

def cadical_baseline(dimacs_path: str) -> Tuple[bool, List[int] | None, Dict]:
    """
    Plain CaDiCaL (no propagator). Returns (sat, model, stats dict).
    Stats include restarts/conflicts/decisions/propagations/conflict_literals(=0)
    and time_ms: parse/simplify(0)/solve/total.
    """
    t0 = time.perf_counter()
    cnf = CNF(from_file=dimacs_path)
    t_parse = time.perf_counter()

    stats = {}
    with Cadical195(bootstrap_with=cnf.clauses) as S:
        t_s0 = time.perf_counter()
        sat = S.solve()
        t_s1 = time.perf_counter()
        model = S.get_model() if sat else None
        try:
            stats = S.accum_stats()
        except Exception:
            stats = {}

    t1 = time.perf_counter()

    stats = dict(stats) if stats else {}
    raw_stats = {
        "restarts": _int_stat(stats, "restarts"),
        "conflicts": _int_stat(stats, "conflicts"),
        "decisions": _int_stat(stats, "decisions"),
        "propagations": _int_stat(stats, "propagations"),
        "conflict_literals": 0,
        "time_ms": {
            "parse": (t_parse - t0) * 1000.0,
            "simplify": 0.0,
            "solve": (t_s1 - t_s0) * 1000.0,
            "total": (t1 - t0) * 1000.0,
        },
    }
    return bool(sat), model, raw_stats


# --------------- guided (ImitSAT Propagator) ---------------

def cadical_imitsat(
        dimacs_path: str,
        runner: ImitSATRunner,
        guide_limit: int,
        phase_policy: str,
) -> Tuple[bool, List[int] | None, Dict]:
    """
    CaDiCaL + ImitSAT Propagator using a *shared* ImitSATRunner (already loaded).
    Returns (sat, model, imitsat_stats dict).
    """
    t0 = time.perf_counter()
    cnf = CNF(from_file=dimacs_path)
    t_parse = time.perf_counter()

    prop = ImitSATPropagator(cnf, runner, guide_limit=guide_limit, phase_policy=phase_policy)

    stats = {}
    with Cadical195(bootstrap_with=cnf.clauses) as S:
        S.connect_propagator(prop)
        for v in range(1, cnf.nv + 1):
            S.observe(v)

        model_ms_before = float(runner.model_time_ms)
        t_s0 = time.perf_counter()
        sat = S.solve()
        t_s1 = time.perf_counter()
        model_ms_after = float(runner.model_time_ms)

        model = S.get_model() if sat else None
        try:
            stats = S.accum_stats()
        except Exception:
            stats = {}

    t1 = time.perf_counter()

    stats = dict(stats) if stats else {}
    imitsat_stats = {
        "restarts": _int_stat(stats, "restarts"),
        "conflicts": _int_stat(stats, "conflicts"),
        "decisions": _int_stat(stats, "decisions"),
        "propagations": _int_stat(stats, "propagations"),
        "conflict_literals": 0,  # keep field for compatibility
        "time_ms": {
            "parse": (t_parse - t0) * 1000.0,
            "simplify": 0.0,
            "solve": (t_s1 - t_s0) * 1000.0,
            "total": (t1 - t0) * 1000.0,
        },
        "model_time_ms": model_ms_after - model_ms_before,
    }
    return bool(sat), model, imitsat_stats


# --------------- per-instance record ---------------

def process_single_cnf_with_runner(
        file_path: str,
        runner: ImitSATRunner,
        guide_limit: int,
        phase_policy: str,
) -> Dict:
    """
    For one DIMACS file:
      1) run baseline CaDiCaL -> raw_stats
      2) run CaDiCaL + ImitSAT -> imitsat_stats
      3) return the compact record required
    """
    cnf_obj = CNF(from_file=file_path)
    cnf_dimacs = cnf_obj.to_dimacs()
    n_v = cnf_obj.nv
    n_c = len(cnf_obj.clauses)

    base_sat, _base_model, raw_stats = cadical_baseline(file_path)

    guided_sat, guided_model, imitsat_stats = cadical_imitsat(
        file_path, runner=runner, guide_limit=guide_limit, phase_policy=phase_policy
    )

    sat_int = int(bool(base_sat))

    rec = {
        "cnf": cnf_dimacs,
        "n_v": n_v,
        "n_c": n_c,
        "satisfiable": sat_int,
        "raw_stats": raw_stats,
        "imitsat_stats": imitsat_stats,
    }
    return rec


# --------------- TXT mode (load once per TXT) ---------------

def process_txt_file(
        txt_file: str,
        save_path: str,
        output_prefix: str,
        model_dir: str,
        model_config: str,
        guide_limit: int,
        phase_policy: str,
) -> None:
    """
    Read single-line CNFs from TXT, load model ONCE, solve all, write ONE JSON.
    """
    problems = read_sat_problems_lines(txt_file)

    if os.path.isdir(save_path) or save_path.endswith(os.sep):
        ensure_dir(save_path)
        out_json = os.path.join(save_path, f"{output_prefix}_{Path(txt_file).stem}.json")
    else:
        ensure_parent_dir(save_path)
        out_json = save_path

    runner = ImitSATRunner(model_dir=model_dir, model_config=model_config)

    results: List[Dict] = []
    tmp_filename = "./_tmp_cadical.cnf"

    try:
        for idx, line in tqdm(list(enumerate(problems)), desc=f"TXT:{Path(txt_file).name}"):
            cnf_formula = cnf_line_2_CNF_class(line)
            write_temp_cnf_file(cnf_formula, filename=tmp_filename)

            rec = process_single_cnf_with_runner(
                tmp_filename, runner=runner, guide_limit=guide_limit, phase_policy=phase_policy
            )
            results.append(rec)
    finally:
        try:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
        except OSError:
            pass

    save_dicts_to_json(results, out_json)
    print(f"[OK] Results saved to {out_json}  (instances: {len(results)})")


# --------------- Folders mode (load once per subfolder) ---------------

def process_folders(
        parent_folder: str,
        save_dir: str,
        output_prefix: str,
        model_dir: str,
        model_config: str,
        guide_limit: int,
        phase_policy: str,
) -> None:
    """
    For each subfolder: load/warmup the ImitSAT model ONCE, process all *.cnf, write ONE JSON per subfolder.
    """
    ensure_dir(save_dir)

    for entry in os.scandir(parent_folder):
        if not entry.is_dir():
            continue

        subfolder = entry.path
        subname = Path(subfolder).name
        out_json = os.path.join(save_dir, f"{output_prefix}_{subname}.json")
        print(f"[INFO] {subname}: loading ImitSAT model once and processing CNFs...")

        runner = ImitSATRunner(model_dir=model_dir, model_config=model_config)

        results: List[Dict] = []
        cnf_files = [
            os.path.join(subfolder, f)
            for f in sorted(os.listdir(subfolder))
            if f.lower().endswith(".cnf")
        ]

        for fpath in tqdm(cnf_files, desc=f"{subname}"):
            try:
                rec = process_single_cnf_with_runner(
                    fpath, runner=runner, guide_limit=guide_limit, phase_policy=phase_policy
                )
                results.append(rec)
            except Exception as e:
                results.append({
                    "cnf": f"(error while reading {os.path.abspath(fpath)})",
                    "n_v": 0,
                    "n_c": 0,
                    "satisfiable": 0,
                    "raw_stats": {"restarts": 0, "conflicts": 0, "decisions": 0, "propagations": 0,
                                  "conflict_literals": 0, "time_ms": {"parse": 0.0, "simplify": 0.0,
                                                                      "solve": 0.0, "total": 0.0}},
                    "imitsat_stats": {"restarts": 0, "conflicts": 0, "decisions": 0, "propagations": 0,
                                      "conflict_literals": 0, "time_ms": {"parse": 0.0, "simplify": 0.0,
                                                                          "solve": 0.0, "total": 0.0},
                                      "model_time_ms": 0.0},
                    "error": f"{type(e).__name__}: {e}",
                })

        save_dicts_to_json(results, out_json)
        print(f"[OK] {subname}: {len(results)} instances → {out_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run CaDiCaL (+ImitSAT) from TXT list or subfolders of .cnf, "
                    "loading the ImitSAT model ONCE per TXT or per subfolder."
    )
    p.add_argument("--mode", choices=["txt", "folders"], required=True)
    p.add_argument("--txt-file", type=str, help="TXT with one-line CNFs (required for --mode txt).")
    p.add_argument("--folder", type=str, help="Parent folder with subfolders of .cnf (required for --mode folders).")

    p.add_argument("--save-path", type=str, default="./output/imitsat_cadical/",
                   help="TXT mode: JSON path (file or dir). Folders mode: output dir.")
    p.add_argument("--output-prefix", type=str, default="ImitSAT_CaDiCaL",
                   help="Prefix for JSON filenames in folders mode.")

    p.add_argument("--model-dir", type=str, default="./model_ckpt/",
                   help="ImitSAT model directory.")
    p.add_argument("--model-config", type=str, default="./model_config/ImitSAT_config.json",
                   help="ImitSAT model config JSON.")

    p.add_argument("--guide-limit", type=int, default=5,
                   help="Number of model-guided decisions before fallback.")
    p.add_argument("--phase-policy", type=str, choices=["save", "pos", "neg", "model"], default="model",
                   help="Polarity policy for guided decisions.")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "txt":
        if not args.txt_file:
            raise SystemExit("--txt-file is required for --mode txt")
        process_txt_file(
            txt_file=args.txt_file,
            save_path=args.save_path,
            output_prefix=args.output_prefix,
            model_dir=args.model_dir,
            model_config=args.model_config,
            guide_limit=args.guide_limit,
            phase_policy=args.phase_policy,
        )
    else:
        if not args.folder:
            raise SystemExit("--folder is required for --mode folders")
        process_folders(
            parent_folder=args.folder,
            save_dir=args.save_path,
            output_prefix=args.output_prefix,
            model_dir=args.model_dir,
            model_config=args.model_config,
            guide_limit=args.guide_limit,
            phase_policy=args.phase_policy,
        )


if __name__ == "__main__":
    main()
