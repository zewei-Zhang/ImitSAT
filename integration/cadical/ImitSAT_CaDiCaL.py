# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
ImitSAT ↔ PySAT (CaDiCaL) integration.

- Loads the JAX/Haiku ImitSAT once
- Builds the same tokenized prefix as your MiniSAT integration:
    "[CNF] <single-line cnf> [SEP]  D d1  D d2  ...  D"
- Implements a PySAT Propagator so CaDiCaL asks us for each decision.
- Returns a literal (±var); return 0 to let CaDiCaL decide.
"""

import re
import time
from typing import List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from pysat.formula import CNF
from pysat.solvers import Cadical195
from pysat.engines import Propagator

from ImitSAT.run_ImitSAT import load_imitsat_model_for_inference

# --------------------------------------------------------------------------------------
# 0) Utilities to mirror your CNF string preparation
# --------------------------------------------------------------------------------------

_INT_START = re.compile(r"\s*([+-]?\d+)\b")


def _is_nonzero_leading_int(line: str) -> bool:
    m = _INT_START.match(line)
    return bool(m and int(m.group(1)) != 0)


def cnf_to_single_line_text(cnf: CNF) -> str:
    """
    Reproduce the same CNF string you build in apply_imitsat_branch:
      - drop 'p cnf' and comments
      - keep only lines starting with nonzero signed int
      - join with spaces into a single line
    """
    dim = cnf.to_dimacs()
    lines = dim.splitlines()[1:]  # skip the 'p cnf' header (first line)
    keep = [ln for ln in lines if _is_nonzero_leading_int(ln)]
    one_line = " ".join(keep).replace("\n", " ")
    return one_line


def prefix_to_dyn_text(decisions: List[int]) -> str:
    """
    Build the dynamic tail " D d1  D d2  ...  D".
    We prepend 'D' before each decision (surviving stack), and add a trailing ' D'
    to match your current MiniSAT path (" ... + ' D' ").
    """
    if not decisions:
        return " D"
    parts = []
    for lit in decisions:
        parts.append(f"D {lit}")
    return " " + " ".join(parts) + " D"


# --------------------------------------------------------------------------------------
# 1) A tiny runner that wraps your model exactly like your MiniSAT path
# --------------------------------------------------------------------------------------

class ImitSATRunner:
    """
    A thin wrapper around your existing model code to:
      - load once (params/apply/tokenizer/argmax)
      - cache the "[CNF] ... [SEP]" token ids per instance
      - produce the next decision variable (1..n) from a signed-decisions prefix
    """

    def __init__(self, model_dir: str, model_config: str, use_gpu: bool = True):
        # Reuse your loader (includes warmup/JIT)
        (self.params,
         self.apply_fn,
         self.tokenizer,
         self.context_len,
         self.latent_len,
         self._argmax_last) = load_imitsat_model_for_inference(model_dir, model_config)

        # Runtime cache
        self._pad_id = int(self.tokenizer.pad_token_id or 0)
        self._cnf_prefix_ids: Optional[np.ndarray] = None
        self._nvars: int = 0
        self._rng = jax.random.PRNGKey(42)

        # Stats (optional)
        self.model_time_ms = 0.0

    def prepare_cnf(self, cnf: CNF):
        """Compute and cache the [CNF]... [SEP] token ids for this instance."""
        self._nvars = cnf.nv
        cnf_str = cnf_to_single_line_text(cnf)
        prefix_text = f"[CNF] {cnf_str} [SEP]"
        self._cnf_prefix_ids = np.asarray(
            self.tokenizer.encode(prefix_text, add_special_tokens=False),
            dtype=np.int32
        )

    def next_decision_from_dyn_text(self, dyn_text: str) -> Tuple[int, int]:
        """
        dyn_text: the exact simplified tail, e.g. " D -1 D -24 9 10 D"
        Returns: (var, sign) where var in [1..nvars], sign in {+1,-1}; or (0, +1) to decline.
        """
        dyn_ids = np.asarray(self.tokenizer.encode(dyn_text, add_special_tokens=False), dtype=np.int32)

        S = int(self.context_len)
        input_ids = np.full((1, S), self._pad_id, dtype=np.int32)

        pref = self._cnf_prefix_ids
        pref_len = min(len(pref), S)
        if pref_len:
            input_ids[0, :pref_len] = pref[:pref_len]
        rem = S - pref_len
        if rem > 0:
            dyn_len = min(len(dyn_ids), rem)
            if dyn_len:
                input_ids[0, pref_len:pref_len + dyn_len] = dyn_ids[:dyn_len]

        t0 = time.perf_counter()
        logits = self.apply_fn(self.params, self._rng, input_ids, 1, False)
        logits = jax.block_until_ready(logits)
        t1 = time.perf_counter()
        self.model_time_ms += (t1 - t0) * 1000.0

        predicted_tokens = self._argmax_last(logits)
        pred_ids = np.asarray(predicted_tokens, dtype=np.int32).reshape(-1)
        decoded_str = self.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=False)
        tok = decoded_str.split()[0] if decoded_str else ""

        try:
            signed = int(tok)  # keep sign
        except Exception:
            return 0, +1

        v = abs(signed)
        if 1 <= v <= self._nvars:
            return v, (+1 if signed > 0 else -1)
        return 0, +1


# --------------------------------------------------------------------------------------
# 2) PySAT Propagator for CaDiCaL
# --------------------------------------------------------------------------------------

class ImitSATPropagator(Propagator):
    def __init__(self, cnf, runner, guide_limit=3, phase_policy="model"):
        super().__init__()
        self.cnf = cnf
        self.runner = runner
        self.runner.prepare_cnf(self.cnf)
        self.nvars = cnf.nv
        self.guide_limit = int(guide_limit)
        self.current_calls = 0
        self.phase_policy = phase_policy  # "model" | "save" | "pos" | "neg"

        self.level = 0
        self.events: List[Tuple[str, int, int]] = []
        self.phase_cache = {}
        self._solver = None
        self._seen_new_level = False

    def setup_observe(self, solver):
        self._solver = solver
        for v in range(1, self.nvars + 1):
            solver.observe(v)

    def on_new_level(self):
        self.level += 1
        self._seen_new_level = True

    def on_backtrack(self, to_level: int):
        self.level = to_level
        while self.events and self.events[-1][2] > to_level:
            self.events.pop()

    def on_assignment(self, lit: int, fixed: Optional[bool] = None):
        is_decision = False
        if self._solver is not None and hasattr(self._solver, "is_decision"):
            try:
                is_decision = bool(self._solver.is_decision(lit))
            except Exception:
                is_decision = False

        if not is_decision and self._seen_new_level:
            is_decision = True

        if self._seen_new_level:
            self._seen_new_level = False

        self.phase_cache[abs(lit)] = +1 if lit > 0 else -1

        if is_decision:
            self.events.append(("D", lit, self.level))
        else:
            self.events.append(("A", lit, self.level))

    def _build_dyn_text(self) -> str:
        toks = []
        for etype, lit, _lvl in self.events:
            if etype == "D":
                toks.append("D");
                toks.append(str(lit))
            else:
                toks.append(str(lit))
        return " " + " ".join(toks) + " D"

    def _num_decisions(self) -> int:
        return sum(1 for e in self.events if e[0] == "D")

    def decide(self) -> int:
        if self.current_calls >= self.guide_limit:
            return 0

        self.current_calls += 1
        dyn_text = self._build_dyn_text()

        var, model_sign = self.runner.next_decision_from_dyn_text(dyn_text)
        if not (1 <= var <= self.nvars):
            return 0

        if self.phase_policy == "pos":
            return +var
        if self.phase_policy == "neg":
            return -var
        if self.phase_policy == "save":
            sign = self.phase_cache.get(var, +1)
            return var if sign > 0 else -var

        return var if model_sign > 0 else -var

    def add_clause(self) -> List[int]:
        return []

    def propagate(self) -> List[int]:
        return []

    def provide_reason(self, lit: int) -> List[int]:
        return []

    def check_model(self, model: List[int]) -> bool:
        return True


# --------------------------------------------------------------------------------------
# 3) Convenience function to run on a DIMACS file
# --------------------------------------------------------------------------------------

def run_cadical_with_imitsat(dimacs_path: str,
                             model_dir: str = "./model_ckpt",
                             model_config: str = "./model_config/ImitSAT_config.json",
                             guide_limit: int = 3,
                             phase_policy: str = "model"):
    cnf = CNF(from_file=dimacs_path)
    runner = ImitSATRunner(model_dir=model_dir, model_config=model_config)
    prop = ImitSATPropagator(cnf, runner, guide_limit=guide_limit, phase_policy=phase_policy)

    with Cadical195(bootstrap_with=cnf.clauses) as S:
        S.connect_propagator(prop)

        for v in range(1, cnf.nv + 1):
            S.observe(v)

        sat = S.solve()
        model = S.get_model() if sat else None
        stats = {}
        try:
            stats = S.accum_stats()
        except Exception:
            pass
    return sat, model, stats
