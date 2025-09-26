"""
This file includes some help functions for KeyTrace.

This module provides small helpers to:
  * convert structured MiniSat events into a flat trace string,
  * simplify traces for training (normalize tokens),
  * extract a compact "key trace" that survives backtracking,
  * pull trace tokens from raw MiniSat stdout,
  * read ordered decision numbers from a trace string.
"""
import re

from typing import List


def simplify_trace(trace_str: str) -> str:
    """
    Unify 'BT' -> 'D', remove 'L x',
    keep the literal after 'A' but remove the 'A' token itself,
    and keep everything else as-is.

    For example:
      "D -1 L 1 BT -24 L 2 A 9 A 10" -> "D -1 D -24 9 10"
    """
    tokens = trace_str.split()
    new_tokens = []
    i = 0

    while i < len(tokens):
        t = tokens[i]

        if t == "BT":
            new_tokens.append("D")
            i += 1
        elif t == "L":
            i += 2
        elif t == "A":
            i += 1
            if i < len(tokens):
                new_tokens.append(tokens[i])
            i += 1
        else:
            new_tokens.append(t)
            i += 1

    return " ".join(new_tokens)


def convert_keytrace_to_str(events):
    """
    Converts a list of tuples like:
        [('BT', 192, 0), ('BT', 96, 0), ('A', 149, 0), ('BT', -155, 0), ('A', 185, 0), ...]
    into a string like:
        "BT 192 L 0 BT 96 L 0 A 149 BT -155 L 0 A 185 A -28 A 59"

    Rules:
    - If etype in ('D', 'BT'): output "<etype> <val> L <lvl>"
    - If etype == 'A': output "A <val>" (ignoring lvl if you store it)
    """
    out_tokens = []
    for etype, val, lvl in events:
        if etype in ("D", "BT"):
            out_tokens.append(etype)
            out_tokens.append(str(val))
            out_tokens.append("L")
            out_tokens.append(str(lvl))
        elif etype == "A":
            out_tokens.append("A")
            out_tokens.append(str(val))
        else:
            pass

    return " ".join(out_tokens)


def extract_trace(minisat_output: str) -> str:
    """
    Extracts the trace from MiniSat output, focusing on 'D', 'A', and 'BT' entries.

    Args:
        minisat_output (str): The output from MiniSat.

    Returns:
        str: The extracted trace as a single string.
    """
    trace_pattern = re.compile(r'(D\s+[-]?\d+\s+L\s+\d+|A\s+[-]?\d+|BT\s+[-]?\d+\s+L\s+\d+|L\s+0)')
    trace_matches = trace_pattern.findall(minisat_output)
    return ' '.join(trace_matches)


def extract_numbers_in_order(trace_string: str) -> List[int]:
    """
    Extracts the numbers in order of traces.

    Args:
        trace_string: A string of traces.

    Returns:
        A list of integers.
    """
    pattern = r'(?:D|BT)\s+(-?\d+)'
    matches = re.findall(pattern, trace_string)
    return [int(m) for m in matches]


def get_key_trace(trace: str) -> str:
    """
    Extract the key trace from the entire trace.

    Args:
        trace: A string represents the entire trace.

    Returns:
        str: The extracted key trace as a single string.
    """
    tokens = trace.split()
    index = 0
    stack = []
    current_level = 0
    while index < len(tokens):
        token = tokens[index]
        if token == 'D':
            index += 1
            var = tokens[index]
            index += 1
            if tokens[index] != 'L':
                raise ValueError("Expected 'L' after decision variable")
            index += 1
            level = int(tokens[index])
            current_level = level

            step = ['D', var, 'L', str(level)]
            stack.append((level, step))
            index += 1
        elif token == 'A':
            index += 1
            var = tokens[index]
            step = ['A', var]
            stack.append((current_level, step))
            index += 1
        elif token == 'BT':
            index += 1
            bt_var = tokens[index]
            index += 1
            if tokens[index] != 'L':
                raise ValueError("Expected 'L' after BT level")
            index += 1
            back_level = int(tokens[index])
            # Remove entries from the stack with levels higher than back_level
            stack = [(lvl, s) for (lvl, s) in stack if lvl <= back_level]

            step = ['BT', bt_var, 'L', str(back_level)]
            stack.append((back_level, step))
            current_level = back_level
            index += 1
        elif token == 'L':
            index += 1
            zero_str = tokens[index]
            if zero_str != '0':
                raise ValueError(f"Expected '0' after L, got '{zero_str}'")
            index += 1

            stack.clear()
            current_level = 0
        else:
            raise ValueError(f"Unknown token '{token}'")

    final_trace = []
    for lvl, step in stack:
        final_trace.extend(step)
    return ' '.join(final_trace)
