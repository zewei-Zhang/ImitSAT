"""
Original MIT License:

Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

--------------------------------------------------------------------------------------------------

Additional modifications and original work by Zewei Zhang.
"""
import sys
import gzip
import time
import psutil
import argparse

from py_minisat22.minisat22 import MinisatSolver


def parse_dimacs_minisat_like(filename, S):
    """
    Reads a CNF file in a Minisat-like manner:
    - Reads line by line.
    - On encountering a 'p cnf' line, just note the variables/clauses count (metadata).
    - For each clause line:
        - Parse each literal.
        - Create variables on-the-fly if needed.
        - Add the clause immediately after finishing reading it.
    """
    open_fn = gzip.open if filename.endswith('.gz') else open
    with open_fn(filename, 'rt', encoding='utf-8') as f:
        vars = 0
        clauses = 0
        clause_count = 0

        lits = []

        for line in f:
            line = line.strip()
            if not line or line[0] in ('c', 'C', '%', '0'):
                continue

            if line[0] == 'p':
                parts = line.split()
                if len(parts) >= 4 and parts[1] == 'cnf':
                    vars = int(parts[2])
                    clauses = int(parts[3])
                else:
                    print("PARSE ERROR! Unexpected format in the 'p' line.")
                    return False
                continue

            parts = line.split()
            lits.clear()
            for lit_str in parts:
                lit_val = int(lit_str)
                if lit_val == 0:
                    clause_count += 1
                    if not S.addClause_(lits):
                        return True
                    break
                else:
                    v = abs(lit_val) - 1
                    while S.nVars() <= v:
                        S.newVar(sign_=True, dvar=True)
                    internal_lit = (v << 1) + (1 if lit_val < 0 else 0)
                    lits.append(internal_lit)

        if vars != 0 and vars != S.nVars():
            print("WARNING! DIMACS header mismatch: wrong number of variables.")
        if clauses != 0 and clause_count != clauses:
            print("WARNING! DIMACS header mismatch: wrong number of clauses.")

    return False


def print_stats(S, start_time):
    cpu_time = time.process_time() - start_time

    # Compute memory used (in MB) - requires psutil:
    # If psutil is not available, you can skip or approximate.
    process = psutil.Process()
    mem_used = process.memory_info().rss / (1024 * 1024)  # in MB

    # Avoid division by zero:
    conflicts_per_sec = S.conflicts / cpu_time if cpu_time > 0 else 0
    decisions_per_sec = S.decisions / cpu_time if cpu_time > 0 else 0
    propagations_per_sec = S.propagations / cpu_time if cpu_time > 0 else 0

    # Compute percentage random decisions:
    random_percent = (S.rnd_decisions * 100 / S.decisions) if S.decisions > 0 else 0.0

    # Compute percentage of conflict literals deleted:
    # If max_literals == tot_literals, then no literals were deleted.
    deleted_percent = ((S.max_literals - S.tot_literals) * 100 / S.max_literals) if S.max_literals > 0 else 0.0

    print("restarts              : {}".format(S.starts))
    print("conflicts             : {:<14} ({:.0f} /sec)".format(S.conflicts, conflicts_per_sec))
    print("decisions             : {:<14} ({:.2f} %% random) ({:.0f} /sec)".format(S.decisions, random_percent, decisions_per_sec))
    print("propagations          : {:<14} ({:.0f} /sec)".format(S.propagations, propagations_per_sec))
    print("conflict literals     : {:<14} ({:.2f} %% deleted)".format(S.tot_literals, deleted_percent))
    print("Memory used           : {:.2f} MB".format(mem_used))
    print("CPU time              : {:.3f} s".format(cpu_time))



def main():
    parser = argparse.ArgumentParser(
        description="Solve a DIMACS CNF with python version Minisat22."
    )
    parser.add_argument(
        "-i", "--input_file",
        default="../dataset/minisat22_test/example.cnf",
        help="Path to input CNF file (.cnf or .cnf.gz)."
    )
    parser.add_argument(
        "-o", "--output_file",
        default="../output/minisat22_py/output.txt",
        help="Path to write result (SAT/UNSAT + model). Use '-' for stdout."
    )
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    start_time = time.process_time()

    S = MinisatSolver()
    S.verbosity = 1
    unsat = parse_dimacs_minisat_like(input_file, S)
    if unsat:
        print("UNSATISFIABLE")
        return

    if not S.simplify():
        print("UNSATISFIABLE")
        if output_file:
            with open(output_file, 'w') as rf:
                rf.write("UNSAT\n")
        sys.exit(20)

    # Solve
    result = S.solve_()  # returns Lbool.TRUE (0), Lbool.FALSE (1), or Lbool.UNDEF (2)

    print_stats(S, start_time)

    if result == 0:
        # SAT
        print("SATISFIABLE")
        if output_file:
            with open(output_file, 'w') as rf:
                rf.write("SAT\n")
                # Convert model back to dimacs form
                model = []
                for i, val in enumerate(S.model):
                    # val: 0=TRUE,1=FALSE,2=UNDEF
                    if val == 0:
                        model.append(str(i + 1))
                    else:
                        model.append(str(-(i + 1)))
                rf.write(" ".join(model) + " 0\n")
        sys.exit(10)
    elif result == 1:
        # UNSAT
        print("UNSATISFIABLE")
        if output_file:
            with open(output_file, 'w') as rf:
                rf.write("UNSAT\n")
        sys.exit(20)
    else:
        # UNDEFINED
        print("INDETERMINATE")
        if output_file:
            with open(output_file, 'w') as rf:
                rf.write("INDET\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
