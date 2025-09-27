# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
ImitSAT–MiniSAT integration.

A branching policy for CDCL that queries an ImitSAT model to propose
the next decision literal.
"""
import jax
import numpy as np
import time
from typing import List
from py_minisat22.minisat22 import MinisatSolver, var, Lbool, sign, CRef
from utils.keytrace_utils import convert_keytrace_to_str, simplify_trace


def compute_next_decision_value(x: int) -> int:
    """
    Map a decision value from trace for minisat.

    Args:
        x (int): Input integer.

    Returns:
        int: Computed next value.
    """
    return 2 * (x - 1) if x > 0 else -2 * x - 1


class ImitSATSolver(MinisatSolver):
    """
    MiniSAT subclass that uses ImitSAT to pick branch literals.
    """
    def __init__(self):
        super().__init__()
        self.use_model = True

        self.imitsat_model_apply_fn = None
        self.imitsat_params = None
        self.imitsat_tokenizer = None
        self._argmax_last = None
        self.inference_rng = jax.random.PRNGKey(42)

        self.context_len = 0
        self.latent_len = 0
        self._pad_id = 0
        self._cnf_prefix_ids = None

        self.cnf_str = None
        self.max_retries = 1
        self.num_imitsat_pick = 0
        self.num_normal_pick = 0

        self.max_imitsat_pick_num = 10
        self.model_time_ms = 0.0
        self.use_model_num = 0

    def set_model_tokenizer(self, apply_fn, params, imitsat_tokenizer, _argmax_last):
        """
        Register the ImitSAT apply function, params, tokenizer, and argmax helper.

        Args:
            apply_fn: JAX/Haiku apply(params, rng, input_ids, latent_len, is_training)->logits.
        	params: ImitSAT model parameters pytree.
    		imitsat_tokenizer: ImitSAT tokenizer with encode/decode and pad_token_id.
    		_argmax_last: Function mapping logits to predicted token ids.
        """
        self.imitsat_tokenizer = imitsat_tokenizer
        self.imitsat_tokenizer.padding_side = 'right'
        self.imitsat_model_apply_fn = apply_fn
        self.imitsat_params = params
        self._argmax_last = _argmax_last
        self.model_time_ms = 0.0

        self._pad_id = int(self.imitsat_tokenizer.pad_token_id or 0)
        prefix_text = f"[CNF] {self.cnf_str} [SEP]"
        self._cnf_prefix_ids = self.imitsat_tokenizer.encode(prefix_text, add_special_tokens=False)

    def set_cnf(self, cnf_str: str):
        """
        Set the CNF text used to build the ImitSAT prefix.

        Args:
        	cnf_str: Single-line CNF content (no headers).
        """
        self.cnf_str = cnf_str

    def addClause_(self, ps: List[int]) -> bool:
        """
        Insert a clause at decision level 0 (MiniSAT internal API).
        """
        assert self.decisionLevel() == 0
        if not self.ok:
            return False

        original_ps = ps[:]

        ps.sort(key=lambda x: (var(x), sign(x)))
        p = self.lit_Undef
        j = 0
        i = 0
        while i < len(ps):
            if self.value(ps[i]) == Lbool.TRUE or (p != self.lit_Undef and ps[i] == (p ^ 1)):
                return True
            elif self.value(ps[i]) != Lbool.FALSE and (p == self.lit_Undef or ps[i] != p):
                ps[j] = ps[i]
                p = ps[i]
                j += 1
            i += 1
        ps = ps[:j]
        if len(ps) == 0:
            self.ok = False
            return False
        elif len(ps) == 1:
            self.uncheckedEnqueue(ps[0])
            self.ok = (self.propagate() == CRef.UNDEF)
            return self.ok
        else:
            cr = self.ca.alloc(ps, False)
            self.clauses.append(cr)
            self.attachClause(cr)

            c = self.ca[cr]
            c.original_lits = original_ps
            return True

    @staticmethod
    def parse_int_if_valid(token: str):
        """
        Return True if token is a valid signed integer.

        Args:
            token: Candidate token string.
        """
        s = token.lstrip('+-')
        if s.isdigit():
            return True
        return False

    def pickBranchLit(self) -> int:
        """
        Pick the next decision literal using ImitSAT when enabled; otherwise fallback.
        """
        if not self.use_model or self.use_model_num >= self.max_imitsat_pick_num:
            self.num_normal_pick += 1
            return self.standard_pickBranchLit()

        if self.nAssigns() == self.nVars():
            return self.lit_Undef

        self.use_model_num += 1

        full_trace_events = self.key_trace_events

        trace_str = convert_keytrace_to_str(full_trace_events)

        trace_str = simplify_trace(trace_str)

        dyn_text = " " + trace_str + " D"
        dyn_ids = self.imitsat_tokenizer.encode(dyn_text, add_special_tokens=False)

        S = int(self.context_len)
        pad_id = self._pad_id

        input_ids = np.full((1, S), pad_id, dtype=np.int32)

        pref = self._cnf_prefix_ids if self._cnf_prefix_ids is not None else []
        pref_len = min(len(pref), S)
        if pref_len:
            input_ids[0, :pref_len] = np.asarray(pref[:pref_len], dtype=np.int32)

        rem = S - pref_len
        if rem > 0:
            dyn_len = min(len(dyn_ids), rem)
            if dyn_len:
                input_ids[0, pref_len:pref_len + dyn_len] = np.asarray(dyn_ids[:dyn_len], dtype=np.int32)

        input_ids_jnp = input_ids

        t_model0 = time.perf_counter()
        logits = self.imitsat_model_apply_fn(self.imitsat_params, self.inference_rng,
                                             input_ids_jnp, 1, False)
        logits = jax.block_until_ready(logits)
        t_model1 = time.perf_counter()
        self.model_time_ms += (t_model1 - t_model0) * 1000.0

        predicted_tokens = self._argmax_last(logits)

        decoded_list = self.imitsat_tokenizer.decode(predicted_tokens, skip_special_tokens=False)
        decoded = decoded_list.split()[0]

        if self.parse_int_if_valid(decoded):
            next_decision_lit = abs(int(decoded))
            var_range = self.nVars()
            if next_decision_lit > var_range:
                self.num_normal_pick += 1
                return self.standard_pickBranchLit()
            if self.assigns[next_decision_lit - 1] != Lbool.UNDEF:
                self.num_normal_pick += 1
                return self.standard_pickBranchLit()

            self.num_imitsat_pick += 1
            return compute_next_decision_value(int(decoded))

        self.num_normal_pick += 1
        return self.standard_pickBranchLit()
