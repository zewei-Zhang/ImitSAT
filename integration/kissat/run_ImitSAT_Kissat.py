# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
Run Kissat (with and without ImitSAT guidance) over CNF instances from a TXT
list or folders of .cnf files, and save per‑instance stats to JSON.
"""

import argparse
import os
import time
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm
from pysat.formula import CNF

from utils.utils import (
    read_sat_problems_lines,
    write_temp_cnf_file,
    cnf_line_2_CNF_class,
    save_dicts_to_json,
)

# ---------------------------------------------------------------------------
# Kissat output parsing
# ---------------------------------------------------------------------------

STAT_PATTERNS = {
    "conflicts": re.compile(r"^c\s+conflicts:\s+(\d+)"),
    "decisions": re.compile(r"^c\s+decisions:\s+(\d+)"),
    "propagations": re.compile(r"^c\s+propagations:\s+(\d+)"),
    "restarts": re.compile(r"^c\s+restarts:\s+(\d+)"),
}

TIME_RE = re.compile(r"^c\s+process-time:\s+.*\s([\d.]+)\s+seconds$")


def parse_kissat_stats(output: str) -> Dict[str, Optional[float]]:
    """
    Parse the 'statistics' / 'resources' section of Kissat output.

    Returns dict with keys:
        conflicts, decisions, propagations, restarts, time_s
    Any missing field defaults to 0 (or None for time_s).
    """
    stats = {
        "conflicts": 0,
        "decisions": 0,
        "propagations": 0,
        "restarts": 0,
        "time_s": None,
    }

    for line in output.splitlines():
        line = line.strip()
        for key, pat in STAT_PATTERNS.items():
            m = pat.match(line)
            if m:
                try:
                    stats[key] = int(m.group(1))
                except ValueError:
                    pass
        m = TIME_RE.match(line)
        if m:
            try:
                stats["time_s"] = float(m.group(1))
            except ValueError:
                pass

    return stats


def make_stats_dict(parsed: Dict[str, Optional[float]], wall_s: float) -> Dict:
    """
    Turn parsed stats + wall-clock time into the JSON 'raw_stats' / 'imitsat_stats' dict.
    """
    if parsed.get("time_s") is not None:
        t_s = float(parsed["time_s"])
    else:
        t_s = float(wall_s)

    t_ms = t_s * 1000.0

    return {
        "restarts": int(parsed.get("restarts", 0) or 0),
        "conflicts": int(parsed.get("conflicts", 0) or 0),
        "decisions": int(parsed.get("decisions", 0) or 0),
        "propagations": int(parsed.get("propagations", 0) or 0),
        "time_ms": {
            "parse": 0.0,
            "simplify": 0.0,
            "solve": t_ms,
            "total": t_ms,
        },
    }


def run_kissat_once(
        cnf_path: str,
        kissat_bin: str,
        use_imitsat: bool,
        imitsat_limit: Optional[int] = None,
        imitsat_port: Optional[int] = None,
        imitsat_timeout: Optional[int] = None,
        imitsat_host: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
) -> Tuple[Optional[bool], Dict]:
    """
    Run Kissat on one CNF and return (satisfiable, stats_dict).

    satisfiable = True  -> SAT (exit code 10)
                 False -> UNSAT (exit code 20)
                 None  -> unknown/error/timeout

    stats_dict has the same shape as 'raw_stats' / 'imitsat_stats' above.
    """
    cmd: List[str] = [kissat_bin, "-s"]

    if extra_args:
        cmd.extend(extra_args)

    if use_imitsat:
        cmd.append("--imitsat=1")
        if imitsat_limit is not None:
            cmd.append(f"--imitsat_limit={imitsat_limit}")
        if imitsat_port is not None:
            cmd.append(f"--imitsat_port={imitsat_port}")
        if imitsat_timeout is not None:
            cmd.append(f"--imitsat_timeout={imitsat_timeout}")
    else:
        cmd.append("--imitsat=0")

    cmd.append(os.path.abspath(cnf_path))

    env = os.environ.copy()
    if imitsat_host:
        env["IMITSAT_HOST"] = imitsat_host

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
        wall_s = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        stats = make_stats_dict(
            {"conflicts": 0, "decisions": 0, "propagations": 0, "restarts": 0, "time_s": timeout_s},
            wall_s=timeout_s or 0.0,
        )
        return None, stats

    sat: Optional[bool]
    if proc.returncode == 10:
        sat = True
    elif proc.returncode == 20:
        sat = False
    else:
        sat = None

    parsed = parse_kissat_stats(proc.stdout)
    stats = make_stats_dict(parsed, wall_s=wall_s)
    return sat, stats


def process_single_sat_problem(
        file_name: str,
        kissat_bin: str,
        imitsat_limit: int,
        imitsat_port: int,
        imitsat_host: str,
        imitsat_timeout: int,
        extra_kissat_args: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
) -> Dict:
    """
    Run baseline Kissat and Kissat+ImitSAT on one CNF file and return a stats record.
    """
    cnf_formula = CNF()
    cnf_formula.from_file(file_name)
    cnf_dimacs = cnf_formula.to_dimacs()

    sat_raw, raw_stats = run_kissat_once(
        cnf_path=file_name,
        kissat_bin=kissat_bin,
        use_imitsat=False,
        imitsat_limit=imitsat_limit,
        imitsat_port=imitsat_port,
        imitsat_timeout=imitsat_timeout,
        imitsat_host=imitsat_host,
        extra_args=extra_kissat_args,
        timeout_s=timeout_s,
    )

    sat_imitsat, imitsat_stats = run_kissat_once(
        cnf_path=file_name,
        kissat_bin=kissat_bin,
        use_imitsat=True,
        imitsat_limit=imitsat_limit,
        imitsat_port=imitsat_port,
        imitsat_timeout=imitsat_timeout,
        imitsat_host=imitsat_host,
        extra_args=extra_kissat_args,
        timeout_s=timeout_s,
    )

    satisfiable: Optional[bool]
    if sat_raw is not None:
        satisfiable = sat_raw
    else:
        satisfiable = sat_imitsat

    stats = {
        "cnf": cnf_dimacs,
        "n_v": cnf_formula.nv,
        "n_c": len(cnf_formula.clauses),
        "satisfiable": satisfiable,
        "raw_stats": raw_stats,
        "imitsat_stats": imitsat_stats,
    }

    return stats


def process_sat_problems_from_lines(
        problems: List[str],
        kissat_bin: str,
        imitsat_limit: int,
        imitsat_port: int,
        imitsat_host: str,
        imitsat_timeout: int,
        extra_kissat_args: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
) -> List[Dict]:
    """
    Processes a list of SAT problems (line-encoded CNFs) and returns stats.
    """
    results: List[Dict] = []
    tmp_filename = "./tmp_problem.cnf"

    for idx, problem_line in tqdm(list(enumerate(problems)), desc="instances"):
        cnf_formula = cnf_line_2_CNF_class(problem_line)
        write_temp_cnf_file(cnf_formula, filename=tmp_filename)

        stats = process_single_sat_problem(
            tmp_filename,
            kissat_bin=kissat_bin,
            imitsat_limit=imitsat_limit,
            imitsat_port=imitsat_port,
            imitsat_host=imitsat_host,
            imitsat_timeout=imitsat_timeout,
            extra_kissat_args=extra_kissat_args,
            timeout_s=timeout_s,
        )
        results.append(stats)

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    return results


def process_txt_file(
        problems_file: str,
        save_dir: str,
        output_json_file_pre: str,
        kissat_bin: str,
        imitsat_limit: int,
        imitsat_port: int,
        imitsat_host: str,
        imitsat_timeout: int,
        extra_kissat_args: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
) -> None:
    """
    Read problems from a TXT file, run Kissat (raw + ImitSAT), and save JSON.
    """
    problems = read_sat_problems_lines(problems_file)
    file_name = Path(problems_file).name.split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
    save_json = os.path.join(save_dir, f"{output_json_file_pre}_{file_name}.json")

    results = process_sat_problems_from_lines(
        problems,
        kissat_bin=kissat_bin,
        imitsat_limit=imitsat_limit,
        imitsat_port=imitsat_port,
        imitsat_host=imitsat_host,
        imitsat_timeout=imitsat_timeout,
        extra_kissat_args=extra_kissat_args,
        timeout_s=timeout_s,
    )

    save_dicts_to_json(results, save_json)
    print(f"Results saved to {save_json}")


def process_single_folder(
        folder_name: str,
        kissat_bin: str,
        imitsat_limit: int,
        imitsat_port: int,
        imitsat_host: str,
        imitsat_timeout: int,
        extra_kissat_args: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
) -> List[Dict]:
    """
    Process all .cnf files in a folder with Kissat and return stats list.
    """
    results: List[Dict] = []
    for fname in tqdm(os.listdir(folder_name), desc=f"{folder_name}"):
        if fname.lower().endswith(".cnf"):
            path = os.path.join(folder_name, fname)
            stats = process_single_sat_problem(
                path,
                kissat_bin=kissat_bin,
                imitsat_limit=imitsat_limit,
                imitsat_port=imitsat_port,
                imitsat_host=imitsat_host,
                imitsat_timeout=imitsat_timeout,
                extra_kissat_args=extra_kissat_args,
                timeout_s=timeout_s,
            )
            results.append(stats)
    return results


def process_folders(
        folders: str,
        save_dir: str,
        output_json_file_pre: str,
        kissat_bin: str,
        imitsat_limit: int,
        imitsat_port: int,
        imitsat_host: str,
        imitsat_timeout: int,
        extra_kissat_args: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
) -> None:
    """
    Process each subfolder of CNF files and write one JSON per subfolder.
    """
    os.makedirs(save_dir, exist_ok=True)

    for entry in os.scandir(folders):
        if entry.is_dir():
            subfolder_name = entry.name
            print(f"Processing subfolder: {subfolder_name}")

            results = process_single_folder(
                entry.path,
                kissat_bin=kissat_bin,
                imitsat_limit=imitsat_limit,
                imitsat_port=imitsat_port,
                imitsat_host=imitsat_host,
                imitsat_timeout=imitsat_timeout,
                extra_kissat_args=extra_kissat_args,
                timeout_s=timeout_s,
            )

            out_json = os.path.join(
                save_dir,
                f"{output_json_file_pre}_{subfolder_name}.json",
            )
            save_dicts_to_json(results, out_json)
            print(f"Results for {subfolder_name} saved to {out_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Kissat (baseline + ImitSAT guidance) over CNF instances "
                    "from a TXT file or over folders of .cnf files."
    )

    p.add_argument(
        "--mode",
        choices=["txt", "folders"],
        required=True,
        help="txt: read problems from a text file; "
             "folders: process each subfolder containing .cnf files.",
    )
    p.add_argument(
        "--txt-file",
        type=str,
        default="./dataset/testset/sat_5_15_5000.txt",
        help="Path to the input TXT file (required for --mode txt).",
    )
    p.add_argument(
        "--folder",
        type=str,
        help="Path to the parent folder containing subfolders (required for --mode folders).",
    )

    p.add_argument(
        "--save-path",
        type=str,
        default="./output/kissat_imitsat/",
        help="Output directory for JSON files (same semantics as in ImitSAT_MiniSAT.py).",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default="KissatImitSAT",
        help="Prefix for JSON filenames.",
    )
    p.add_argument(
        "--kissat-bin",
        type=str,
        default="./build/kissat",
        help="Path to the Kissat binary built with ImitSAT integration.",
    )
    p.add_argument(
        "--imitsat-limit",
        type=int,
        default=3,
        help="Max number of guided decisions (--imitsat_limit for Kissat).",
    )
    p.add_argument(
        "--imitsat-port",
        type=int,
        default=8765,
        help="ImitSAT server port (--imitsat_port for Kissat).",
    )
    p.add_argument(
        "--imitsat-host",
        type=str,
        default="127.0.0.1",
        help="ImitSAT server host (exported as IMITSAT_HOST for Kissat).",
    )
    p.add_argument(
        "--imitsat-timeout",
        type=int,
        default=200,
        help="ImitSAT socket timeout in ms (--imitsat_timeout for Kissat).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional per-instance wall-clock timeout in seconds (for Kissat process).",
    )
    p.add_argument(
        "--extra-kissat-args",
        type=str,
        default="",
        help="Extra arguments to pass verbatim to Kissat, e.g. '--seed=1 --quiet=0'.",
    )
    p.add_argument(
        "--lucky",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override Kissat --lucky option (0 or 1).",
    )
    p.add_argument(
        "--probe",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override Kissat --probe option (0 or 1).",
    )

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "txt":
        if not args.txt_file:
            parser.error("--txt-file is required when --mode txt")

        extra_args: List[str] = []
        if args.extra_kissat_args:
            extra_args.extend(args.extra_kissat_args.split())
        if args.lucky is not None:
            extra_args.append(f"--lucky={args.lucky}")
        if args.probe is not None:
            extra_args.append(f"--probe={args.probe}")

        process_txt_file(
            problems_file=args.txt_file,
            save_dir=args.save_path,
            output_json_file_pre=args.output_prefix,
            kissat_bin=args.kissat_bin,
            imitsat_limit=args.imitsat_limit,
            imitsat_port=args.imitsat_port,
            imitsat_host=args.imitsat_host,
            imitsat_timeout=args.imitsat_timeout,
            extra_kissat_args=extra_args,
            timeout_s=args.timeout,
        )
    else:
        if not args.folder:
            parser.error("--folder is required when --mode folders")

        extra_args: List[str] = []
        if args.extra_kissat_args:
            extra_args.extend(args.extra_kissat_args.split())
        if args.lucky is not None:
            extra_args.append(f"--lucky={args.lucky}")
        if args.probe is not None:
            extra_args.append(f"--probe={args.probe}")

        process_folders(
            folders=args.folder,
            save_dir=args.save_path,
            output_json_file_pre=args.output_prefix,
            kissat_bin=args.kissat_bin,
            imitsat_limit=args.imitsat_limit,
            imitsat_port=args.imitsat_port,
            imitsat_host=args.imitsat_host,
            imitsat_timeout=args.imitsat_timeout,
            extra_kissat_args=extra_args,
            timeout_s=args.timeout,
        )


if __name__ == "__main__":
    main()
