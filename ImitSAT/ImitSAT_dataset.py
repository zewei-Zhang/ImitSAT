# *************************************************************************
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
The dataset for ImitSAT. The ImitSATDataset reads CNF items from .jsonl.gz file.
Each item includes z(F,K) and the corresponding target.
"""
import gzip
import json
import random

from torch.utils.data import Dataset

class ImitSATDataset(Dataset):
    """
    Dataset class that loads CNF + KeyTrace from .jsonl.gz lines.
    """

    def __init__(
            self,
            file_path,
            tokenizer,
            context_len=512,
            permute_vars=True,
            max_vid=100,
            latent_len=1
    ):
        """Initializes the dataset.

        Args:
            file_path (str): Path to the JSON lines file.
            tokenizer (ImitSATTokenizer): The custom tokenizer.
            max_context (int): Max token length.
            permute_vars (bool): If True, permute variable IDs.
            max_vid (int): Maximum variable ID for permutation.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.permute_vars = permute_vars
        self.max_vid = max_vid
        self.latent_len = latent_len
        self.d_token = "D"
        self.sep_token = "[SEP]"
        self.d_token_id = self.tokenizer._convert_token_to_id(self.d_token)
        self.sep_token_id = self.tokenizer._convert_token_to_id(self.sep_token)
        self.cnf_tok_id = self.tokenizer._convert_token_to_id("[CNF]")
        self.pad_id = self.tokenizer.pad_token_id
        self.data = []
        self.permute_prob = 1.0
        self.old_padding_side = self.tokenizer.padding_side
        with gzip.open(file_path, "rt") as data:
            for raw in data:
                line = json.loads(raw)
                n_v = line["n_v"]
                cnf = line["cnf"].splitlines()[1:]
                cnf = " ".join(cnf)
                key_trace = line["key_trace"]
                key_trace = self.simplify_trace(key_trace)

                trace_tokens = key_trace.split()
                d_positions = [
                    i for i, tok in enumerate(trace_tokens)
                    if tok == self.d_token
                ]
                for d_i in range(len(d_positions)):
                    d_pos = d_positions[d_i]
                    if d_pos + 1 >= len(trace_tokens):
                        continue
                    next_lit = trace_tokens[d_pos + 1]
                    next_lit = "".join(next_lit)

                    partial_trace = trace_tokens[: d_pos]
                    partial_trace_str = " ".join(partial_trace)

                    self.data.append({
                        "cnf": cnf,
                        "trace": partial_trace_str,
                        "n_v": n_v,
                        "label_lit": next_lit
                    })

    @staticmethod
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

    def __len__(self):
        """Returns dataset length."""
        return len(self.data)

    @staticmethod
    def constant_permuter(line, permutation):
        def remap_token(tok: str) -> str:
            negative = tok.startswith('-')
            var_str = tok[1:] if negative else tok
            if var_str.isdigit():
                old_id = int(var_str)
                if 1 <= old_id < len(permutation):
                    new_id = permutation[old_id]
                    return f"-{new_id}" if negative else str(new_id)
            return tok

        return " ".join(remap_token(t) for t in line.split())

    def __getitem__(self, idx):
        """Fetches one example and processes it into (input_ids, labels, attention_mask)."""
        row = self.data[idx]
        cnf_str = row["cnf"]
        trace_str = row["trace"]
        label_lit = row["label_lit"]
        n_v = row["n_v"]
        if self.permute_vars and random.random() < self.permute_prob:
            cap = n_v + 1
            constants = list(range(1, cap))
            permutation = random.sample(constants, len(constants))
            permutation = [0] + permutation
            cnf_str = self.constant_permuter(cnf_str, permutation)
            trace_str = self.constant_permuter(trace_str, permutation)
            label_lit = self.constant_permuter(label_lit, permutation)

        self.tokenizer.padding_side = 'right'
        cnf_enc = self.tokenizer(f'[CNF] {cnf_str} [SEP] {trace_str} D',
                                 truncation=True, padding="max_length",
                                 max_length=self.context_len, padding_side='right',
                                 return_tensors="pt")

        cnf_ids = cnf_enc["input_ids"].squeeze(0)

        self.tokenizer.padding_side = 'right'
        label_enc = self.tokenizer(f"{label_lit}", truncation=True, padding="max_length",
                                   max_length=self.latent_len, return_tensors="pt", padding_side='right')

        label_ids = label_enc["input_ids"].squeeze(0)
        self.tokenizer.padding_side = self.old_padding_side

        input_ids = cnf_ids

        labels = label_ids

        return {
            "input_ids": input_ids,
            "labels": labels
        }