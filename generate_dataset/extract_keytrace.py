# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
Extract MiniSAT key traces for many CNFs and write gzip‑compressed JSONL.

Example:
    python -m generate_dataset.extract_keytrace \
        --raw-dir ./dataset/train_raw/ \
        --out-dir ./dataset/train/

Output JSONL (gzip):
    {"cnf":"...","n_v":18,"n_c":74,"key_trace":"D -1 L 1 A 9 ..."}
"""

import argparse, json, gzip, os, tempfile
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict

from tqdm import tqdm
from pysat.formula import CNF
from py_minisat22.minisat22 import MinisatSolver
from py_minisat22.run_minisat import parse_dimacs_minisat_like
from utils.keytrace_utils import convert_keytrace_to_str
from utils.utils import write_temp_cnf_file, cnf_line_2_CNF_class


def _solve_and_trace(dimacs_line: str) -> Dict:
    """
    Solve one CNF line with MiniSAT and extract its key trace.

    Args:
    	dimacs_line: One‑line DIMACS‑lite CNF.

    Returns:
    	Record dict with fields: cnf (full DIMACS), n_v, n_c, key_trace.
    """
    cnf_obj = cnf_line_2_CNF_class(dimacs_line)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".cnf", delete=False) as tmp:
        tmp_name = tmp.name
    write_temp_cnf_file(cnf_obj, filename=tmp_name)

    solver = MinisatSolver()
    solver.verbosity = 0
    parse_dimacs_minisat_like(tmp_name, solver)
    solver.simplify()
    solver.solve_()

    trace_str = convert_keytrace_to_str(solver.key_trace_events)
    cnf_obj = CNF(from_string=dimacs_line)

    os.unlink(tmp_name)

    return {
        "cnf": cnf_obj.to_dimacs(),
        "n_v": cnf_obj.nv,
        "n_c": len(cnf_obj.clauses),
        "key_trace": trace_str,
    }


def _process_one_file(raw_path: Path, out_path: Path, workers: int) -> None:
    """
    Process one .txt bucket into a .jsonl.gz of key‑trace records.

    Args:
    	raw_path: Path to input .txt (one CNF per line).
    	out_path: Destination .jsonl.gz path.
    	workers: MiniSAT processes to use in the pool.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open() as fh:
        lines = fh.readlines()

    with Pool(processes=workers) as pool, \
            gzip.open(out_path, "wt") as gz_out:
        for rec in tqdm(pool.imap_unordered(_solve_and_trace, lines, 128),
                        total=len(lines),
                        desc=f"[{raw_path.name}]"):
            gz_out.write(json.dumps(rec, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch key-trace extractor")
    ap.add_argument("--raw-dir", default='../dataset/train_raw/',
                    help="root directory that contains *.txt buckets")
    ap.add_argument("--out-dir", default='../dataset/train/',
                    help="root for *.jsonl.gz (default: alongside raw file)")
    ap.add_argument("--glob", default="**/*.txt",
                    help="glob relative to --raw-dir (default: **/*.txt)")
    ap.add_argument("--workers", type=int, default=cpu_count() // 4,
                    help="MiniSat processes per file.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_dir).resolve()
    out_root = Path(args.out_dir).resolve() if args.out_dir else raw_root

    txt_files = sorted(raw_root.glob(args.glob))
    if not txt_files:
        print("No matching .txt files found.", file=sys.stderr)
        sys.exit(1)

    for raw_path in txt_files:
        rel = raw_path.relative_to(raw_root).with_suffix(".jsonl.gz")
        out_path = out_root / rel
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        print(f"→ {rel}")
        _process_one_file(raw_path, out_path, args.workers)


if __name__ == "__main__":
    main()