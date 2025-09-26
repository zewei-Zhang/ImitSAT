# *************************************************************************
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
Unit test for the Python and C++ versions of MiniSAT
to verify whether their entire execution traces are identical.
"""
import unittest
import os

from utils.utils import (get_cnf_files, run_minisat,
                         read_sat_problems_lines, cnf_line_2_CNF_class, write_temp_cnf_file)
from utils.keytrace_utils import extract_trace
from py_minisat22.minisat22 import MinisatSolver
from py_minisat22.run_minisat import parse_dimacs_minisat_like


class TestTraceComparison(unittest.TestCase):
    def setUp(self):
        self.cnf_folder = '../dataset/minisat22_test'
        self.minisat_c_path = '../py_minisat22/minisat.exe'
        self.cnf_files = get_cnf_files(self.cnf_folder)

    @staticmethod
    def compare_traces(minisat_trace: str, s_trace: str) -> bool:
        """Compares the traces from Minisat and the second solver."""
        return minisat_trace.strip() == s_trace.strip()

    def run_and_get_traces(self, cnf_path: str):
        """
        Runs both solvers on the given CNF path and returns their traces.
        """
        minisat_output = run_minisat(cnf_path, self.minisat_c_path)
        minisat_trace = extract_trace(minisat_output)

        solver = MinisatSolver()
        solver.verbosity = 0
        solver.record_entire_trace = True

        parse_dimacs_minisat_like(cnf_path, solver)
        solver.simplify()
        solver.solve_()
        return minisat_trace, solver.trace

    def test_trace_comparison(self):
        for cnf_file in self.cnf_files:
            with self.subTest(cnf_file=cnf_file):
                minisat_trace, py_solver_trace = self.run_and_get_traces(cnf_file)
                self.assertTrue(
                    self.compare_traces(minisat_trace, py_solver_trace),
                    f"Traces differ for file: {cnf_file}"
                )

    def test_generate_sat_problems(self):
        problems = read_sat_problems_lines('../dataset/testset/sat_5_15_5000.txt')
        tmp_filename = './temp_problem.cnf'
        for index, problem_line in enumerate(problems):
            with self.subTest():
                cnf_formula = cnf_line_2_CNF_class(problem_line)
                write_temp_cnf_file(cnf_formula, filename=tmp_filename)

                minisat_trace, py_solver_trace = self.run_and_get_traces(tmp_filename)

                os.remove(tmp_filename)

                self.assertTrue(
                    self.compare_traces(minisat_trace, py_solver_trace),
                    f"Traces differ for file: {index}"
                )



if __name__ == '__main__':
    unittest.main()
