"""
Generate random 3‑SAT formulas in variable‑count buckets and write each CNF on one line
(DIMACS‑lite: integers with trailing 0 per clause).

Example:
    # One bucket 5–15 with 500 items
    python ./generate_dataset/gen_cnf_buckets.py \
        --vars-min 5 --vars-max 15 --samples 500 --out-dir ./dataset/train_raw/
"""
import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import trange


LITERAL_MAKES_CLAUSE_TRUE_PROB = 0.5


def generate_random_assignment(num_vars: int) -> Dict[str, bool]:
    """
    Generate a random boolean assignment x1..xN.

    Args:
    	num_vars: Number of variables.

    Returns:
    	Mapping from variable name (e.g., "x3") to True/False.
    """
    values = np.random.choice([True, False], size=num_vars)
    return {f"x{i + 1}": values[i] for i in range(num_vars)}


def generate_sat_clause_from_assignment(
        assignment: Dict[str, bool],
        num_literals: int
) -> List[str]:
    """
    Create one clause that is satisfied by the given assignment.

    Args:
    	assignment: Variable → truth value.
    	num_literals: Number of literals in the clause.

    Returns:
    	List of literal strings like ["x1", "-x3", "x7"].
    """
    vars_list = list(assignment.keys())
    selected_vars = random.sample(vars_list, k=num_literals)
    clause = []
    clause_is_true = False

    for i, var in enumerate(selected_vars):
        make_true = (random.random() < LITERAL_MAKES_CLAUSE_TRUE_PROB)
        is_last = (i == num_literals - 1)
        if make_true or (is_last and not clause_is_true):
            literal = var if assignment[var] else f"-{var}"
            clause.append(literal)
            clause_is_true = True
        else:
            literal = f"-{var}" if assignment[var] else var
            clause.append(literal)

    return clause


def adjust_clauses_to_include_unused_vars(
        clauses: List[List[str]],
        assignment: Dict[str, bool],
        unused_vars: List[str],
        clause_size: int
) -> None:
    """
    Ensure every variable appears at least once by patching clauses in place.

    Args:
    	clauses: List of clauses (each a list of literal strings).
    	assignment: Variable → truth value.
    	unused_vars: Variables not yet present in any clause.
    	clause_size: Clause length for replacement policy.
    """
    used_var_count = {}
    for clause in clauses:
        for lit in clause:
            v = lit.lstrip("-")
            used_var_count[v] = used_var_count.get(v, 0) + 1

    for v in unused_vars:
        valid_clause_indices = []
        for idx, clause in enumerate(clauses):
            can_replace = True
            for lit in clause:
                var_name = lit.lstrip("-")
                if used_var_count[var_name] <= 1:
                    can_replace = False
                    break
            if can_replace:
                valid_clause_indices.append(idx)

        if not valid_clause_indices:
            raise RuntimeError("No clause found to safely replace a variable with an unused variable.")

        chosen_idx = random.choice(valid_clause_indices)
        chosen_clause = clauses[chosen_idx]
        drop_idx = random.randrange(clause_size)
        dropped_lit = chosen_clause.pop(drop_idx)
        drop_var_name = dropped_lit.lstrip("-")
        used_var_count[drop_var_name] -= 1

        if assignment[v]:
            chosen_clause.append(v)
        else:
            chosen_clause.append(f"-{v}")

        used_var_count[v] = used_var_count.get(v, 0) + 1


def generate_sat_problem(
        num_vars: int,
        num_clauses: int,
        clause_size: int
) -> List[List[str]]:
    """
    Generate a satisfiable CNF with the requested size.

    Args:
    	num_vars: Number of variables.
    	num_clauses: Number of clauses.
    	clause_size: Literals per clause (e.g., 3 for 3‑SAT).

    Returns:
    	List of clauses (each a list of literal strings).
    """
    assignment = generate_random_assignment(num_vars)
    clauses = [
        generate_sat_clause_from_assignment(assignment, clause_size)
        for _ in range(num_clauses)
    ]

    used_vars = set(lit.lstrip("-") for clause in clauses for lit in clause)
    all_vars = set(assignment.keys())
    unused_vars = list(all_vars - used_vars)

    if unused_vars:
        adjust_clauses_to_include_unused_vars(
            clauses,
            assignment,
            unused_vars,
            clause_size
        )

    return clauses


def to_dimacs_like_format(clauses: List[List[str]]) -> str:
    """
    Convert clauses like ['x1','-x3'] to '1 -3 0' form in a single line.

    Args:
    	clauses: List of clauses (each a list of literal strings).

    Returns:
    	One‑line DIMACS‑lite string with '0' delimiters.
    """
    parts = []
    for clause in clauses:
        c_str = " ".join(
            lit.replace("x", "") if not lit.startswith("-")
            else "-" + lit.replace("-x", "")
            for lit in clause
        )
        parts.append(f"{c_str} 0")
    return " ".join(parts)


def write_bucket(
        vars_min: int,
        vars_max: int,
        samples: int,
        ratio_min: float,
        ratio_max: float,
        clause_size: int,
        out_dir: Path,
) -> None:
    """
    Write a bucket of random CNFs to a text file (one CNF per line).

    Args:
    	vars_min: Inclusive lower bound on #vars.
    	vars_max: Inclusive upper bound on #vars.
    	samples: Number of CNFs to generate.
    	ratio_min: Min clause/var ratio.
    	ratio_max: Max clause/var ratio.
    	clause_size: Literals per clause.
    	out_dir: Output folder for the bucket file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"sat_{vars_min}_{vars_max}.txt"
    with fname.open("w") as fh:
        desc = f"[{fname.name}] {samples:,} formulas"
        for _ in trange(samples, desc=desc):
            n_vars = random.randint(vars_min, vars_max)
            ratio = random.uniform(ratio_min, ratio_max)
            n_clauses = int(round(ratio * n_vars))

            clauses = generate_sat_problem(n_vars, n_clauses, clause_size)
            fh.write(to_dimacs_like_format(clauses) + "\n")



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate bucketed 3-SAT dataset")

    ap.add_argument("--vars-min", type=int, required=True)
    ap.add_argument("--vars-max", type=int, required=True)

    g = ap.add_mutually_exclusive_group(required=True)

    g.add_argument("--samples", type=int,
                   help="Base count for linear / power mode")

    ap.add_argument("--ratio-min", type=float, default=4.1)
    ap.add_argument("--ratio-max", type=float, default=4.4)
    ap.add_argument("--clause-size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    write_bucket(
        vars_min=args.vars_min,
        vars_max=args.vars_max,
        samples=args.samples,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        clause_size=args.clause_size,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
