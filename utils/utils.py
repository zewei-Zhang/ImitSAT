"""
This file includes some general help functions.
"""
import os
import subprocess
import json
import glob

from typing import List
from pysat.formula import CNF


def get_cnf_files(folder_path: str) -> List[str]:
    """Returns a list of .cnf files in the specified folder."""
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.cnf')
    ]


def save_dicts_to_json(results: list, output_filename: str) -> None:
    """
    Saves the results to a JSON file.

    Args:
        results (list): The results to save.
        output_filename (str): The filename for the output JSON file.
    """
    with open(output_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def merge_chunk_files(json_folder: str, merged_output: str) -> None:
    """Merge all JSON files in the specified folder into a single JSON array.

    Args:
        json_folder (str): Path to the folder containing JSON files.
        merged_output (str): File path for the merged JSON output.
    """
    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    merged_data = []

    for file_path in json_files:
        print(f"Merging {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as file_handle:
            file_data = json.load(file_handle)
        merged_data.extend(file_data)

    with open(merged_output, "w", encoding="utf-8") as output_file:
        json.dump(merged_data, output_file, indent=4)

    print(f"Merged {len(json_files)} JSON files into {merged_output}.")


def read_sat_problems_lines(filename: str) -> List[str]:
    """
    Reads SAT problems from a file, each line is a problem in CNF format, e.g., "4 5 -1 0 5 1 -2 0".

    Args:
        filename (str): Path to the file containing SAT problems.

    Returns:
        list: A list of SAT problems, one per line.
    """
    with open(filename, 'r') as file:
        problems = file.readlines()
    return [line.strip() for line in problems if line.strip()]


def write_temp_cnf_file(cnf_formula: CNF, filename: str='./temp_problem.cnf') -> None:
    cnf_formula.to_file(filename)


def cnf_line_2_CNF_class(problem_line: str) -> CNF:
    """
    Parses a problem string into a CNF object.

    Args:
        problem_line (str): The problem string where clauses are divided by '0'.

    Returns:
        CNF object representing the SAT problem.
    """
    cnf = CNF()
    tokens = problem_line.strip().split()
    clause = []
    for token in tokens:
        if token == '0':
            if clause:
                cnf.append(clause)
                clause = []
        else:
            literal = int(token)
            clause.append(literal)

    if clause:
        cnf.append(clause)
    return cnf


def run_minisat(cnf_filename: str, minisat_path: str='minisat.exe') -> str:
    """
    Runs MiniSat with the -trace option on the given CNF file and captures its output.

    Args:
        minisat_path: The path of the MiniSat executable.
        cnf_filename (str): Path to the CNF file to process.

    Returns:
        str: Combined output of MiniSat (stdout and stderr).
    """
    command = [minisat_path, '-trace', cnf_filename]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr
    except Exception as ex:
        return f"Error running MiniSat: {ex}"
