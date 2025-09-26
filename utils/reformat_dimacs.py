"""
Standardize DIMACS CNF files.

This script reads CNF-like inputs and writes clean DIMACS outputs:
    - Drops inline '%' comments.
    - Keeps only leading 'c' comment lines.
    - Recomputes the problem line (`p cnf <vars> <clauses>`).
    - Handles multi-line clauses and missing terminal zeros.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List


def _parse_problem_line(s: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a DIMACS problem line.

    Args:
        s: A line potentially starting with 'p cnf'.

    Returns:
        Tuple (n_vars, n_clauses) if parsed; otherwise (None, None).
    """
    s = s.strip()
    if not s.startswith('p'):
        return (None, None)
    parts = s.split()
    if len(parts) >= 4 and parts[0] == 'p' and parts[1].lower() == 'cnf':
        try:
            return int(parts[2]), int(parts[3])
        except ValueError:
            pass
    return (None, None)


def normalize_dimacs_file(
        in_path: Path,
        out_path: Path,
        *,
        stop_after_declared_clauses: bool = True,
        preserve_leading_comments: bool = True
) -> Tuple[int, int]:
    """
    Normalize a single CNF file into strict DIMACS.

    Args:
        in_path: Input CNF path.
        out_path: Output DIMACS path (will be created).
        stop_after_declared_clauses: If True, stop after declared clauses.
        preserve_leading_comments: If True, keep only initial 'c' comments.

    Returns:
        Tuple (n_vars, n_clauses) written to the output file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clauses_out: List[str] = []
    current_clause: List[str] = []
    max_var = 0
    clause_count = 0
    expected_clauses: Optional[int] = None
    header_comments: List[str] = []
    in_leading_comment_block = True

    with in_path.open('r', encoding='utf-8', errors='ignore') as fin:
        for raw in fin:
            # Drop inline '%' comments anywhere
            if '%' in raw:
                raw = raw.split('%', 1)[0]
            line = raw.strip()
            if not line:
                continue

            lstrip = line.lstrip()
            # Preserve only LEADING comment lines
            if lstrip.startswith('c'):
                if preserve_leading_comments and in_leading_comment_block:
                    header_comments.append(line)
                continue

            # Parse problem line but don't emit it
            if lstrip.startswith('p'):
                in_leading_comment_block = False
                _, nc = _parse_problem_line(line)
                expected_clauses = nc
                continue

            in_leading_comment_block = False

            # Parse clause tokens
            for tok in line.split():
                if tok == '0':
                    # End of clause
                    if current_clause:
                        clauses_out.append(' '.join(current_clause) + ' 0')
                        clause_count += 1
                        current_clause.clear()
                    else:
                        # Lone '0' → empty clause, only if still within declared count (or no declaration)
                        if expected_clauses is None or clause_count < expected_clauses:
                            clauses_out.append('0')
                            clause_count += 1
                    # Stop early if we've reached declared clause count
                    if stop_after_declared_clauses and expected_clauses is not None and clause_count >= expected_clauses:
                        current_clause.clear()
                        for _ in fin:  # exhaust file
                            pass
                        break
                else:
                    # Literal
                    try:
                        lit = int(tok)
                    except ValueError:
                        continue  # ignore junk tokens
                    if lit != 0:
                        max_var = max(max_var, abs(lit))
                        current_clause.append(str(lit))

        # Flush final unterminated clause
        if current_clause:
            clauses_out.append(' '.join(current_clause) + ' 0')
            clause_count += 1
            current_clause.clear()

    with out_path.open('w', encoding='utf-8', errors='ignore') as fout:
        for c in header_comments:
            fout.write(c.rstrip() + '\n')
        # fout.write('c normalized to DIMACS by reformat_dimacs.py\n')
        fout.write(f'p cnf {max_var} {clause_count}\n')
        for cl in clauses_out:
            fout.write(cl.rstrip() + '\n')

    return max_var, clause_count


def normalize_folder(input_dir: Path, output_dir: Path, recursive: bool = True) -> None:
    """
    Walk input_dir, find *.cnf files, and write normalized files to output_dir
    with the SAME filenames and mirrored subfolders.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")

    pattern = '**/*.cnf' if recursive else '*.cnf'
    files = sorted(input_dir.glob(pattern))

    if not files:
        print("No .cnf files found.")
        return

    for f in files:
        rel = f.relative_to(input_dir)
        out_path = (output_dir / rel)  # same name, same extension
        try:
            v, c = normalize_dimacs_file(f, out_path)
            print(f"[OK] {f} -> {out_path}  (vars={v}, clauses={c})")
        except Exception as e:
            print(f"[ERROR] {f}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Normalize all .cnf files from a folder into another, keeping filenames.")
    ap.add_argument("--input_dir", type=Path, default='./dataset/public_dataset/folder_need_to_normalize/',
                    help="Folder containing .cnf files")
    ap.add_argument("--output_dir", type=Path, default='./dataset/public_dataset/folder_normalized',
                    help="Destination folder (will mirror structure)")
    ap.add_argument("-nR", "--no-recursive", action="store_true", help="Do not search subfolders")
    args = ap.parse_args()

    normalize_folder(args.input_dir, args.output_dir, recursive=not args.no_recursive)


if __name__ == "__main__":
    main()
