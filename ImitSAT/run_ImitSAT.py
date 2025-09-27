# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
Run ImitSAT over CNF instances from a TXT list or folders of .cnf files, and save per‑instance stats to JSON.
"""
import argparse
import os
import time
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "bfloat16")
DTYPE = jnp.bfloat16
import numpy as np
import haiku as hk

from typing import List, Dict, Iterable
from tqdm import tqdm
from pysat.formula import CNF

from py_minisat22.minisat22 import MinisatSolver
from py_minisat22.run_minisat import parse_dimacs_minisat_like
from utils.utils import (read_sat_problems_lines, write_temp_cnf_file,
                         cnf_line_2_CNF_class, save_dicts_to_json)
from utils.keytrace_utils import extract_numbers_in_order, convert_keytrace_to_str

from ImitSAT.ImitSAT_MiniSAT import ImitSATSolver
from ImitSAT.ImiSAT_tokenizer import ImitSATTokenizer
from perceiver_ar.perceiver_ar_model import PerceiverAR


def is_nonzero_leading_int(text: str) -> bool:
    """
    Check whether a string starts with a nonzero signed integer.

    Args:
        text: Input line.

    Returns:
        True if, after optional leading whitespace, the line begins with a
        signed integer whose value is not zero; otherwise False.
    """
    match = re.match(r"\s*([+-]?\d+)\b", text)
    if not match:
        return False
    return int(match.group(1)) != 0


def filter_lines_starting_with_nonzero_int(lines: Iterable[str]) -> List[str]:
    """
    Filter lines that start with a nonzero signed integer.

    Args:
        lines: Iterable of input lines.

    Returns:
        Lines whose first token (ignoring leading spaces) is a nonzero
        signed integer.
    """
    return [line for line in lines if is_nonzero_leading_int(line)]


def build_forward_fn(vocab_size, max_seq_len, num_channels, num_heads,
                     num_transformers, dropout_prob, cross_attend_widening_factor, transformer_widening_factor,
                     position_encoding_type, pad_id):
    """
    Create a function that constructs and applies PerceiverAR for ImitSAT.

    Args:
    	vocab_size: Token vocabulary size.
    	max_seq_len: Maximum context length.
    	num_channels: Model channel width.
    	num_heads: Attention heads.
    	num_transformers: Transformers per block.
    	dropout_prob: Dropout probability.
    	cross_attend_widening_factor: MLP factor in cross‑attend.
    	transformer_widening_factor: MLP factor in transformer.
    	position_encoding_type: Positional encoding type.
    	pad_id: Padding token id.
    """

    def forward_fn(input_ids, latent_len, is_training=True):
        """
        Apply PerceiverAR and return logits.

        Args:
        	input_ids: [batch, seq_len] integer tokens.
        	latent_len: Latent length for cross‑attention.
        	is_training: Whether to enable training behavior.
        """
        model = PerceiverAR(
            num_classes=vocab_size,
            input_idx_size=[-1],
            max_context_length=max_seq_len,
            input_embed_dim=num_channels,
            num_z_channels=num_channels,
            num_cross_attend_heads=num_heads,
            num_transformers_per_block=num_transformers,
            num_transformer_heads=num_heads,
            dropout_prob=dropout_prob,
            transformer_widening_factor=transformer_widening_factor,
            cross_attend_widening_factor=cross_attend_widening_factor,
            position_encoding_type=position_encoding_type,
            mask_style='final_block',
            max_wavelength=max_seq_len,
        )
        out = model(
            inputs=input_ids,
            input_idxs=None,
            is_training=is_training,
            memory_type="none",
            memory=None,
            z_index_dim=latent_len if latent_len > 0 else 1,
            use_remat=False
        )
        return out.input_events_logits

    return forward_fn


def apply_key_trace(tmp_filename: str, key_trace: str, key_decision: List[int] = None):
    """
    Replay a key trace with MiniSAT and collect KeyTrace stats.

    Args:
        tmp_filename: Path to DIMACS file.
        key_trace: Trace string from baseline solve.
        key_decision: Optional explicit decision sequence.
    """
    solver_ = MinisatSolver()
    solver_.verbosity = 0

    key_decision = extract_numbers_in_order(key_trace) if key_decision is None else key_decision
    key_decision.reverse()
    solver_.use_keytrace = True
    solver_.keytrace_instructions = key_decision
    if len(key_decision) == 0:
        solver_.use_keytrace = False

    t0 = time.perf_counter()
    unsat = parse_dimacs_minisat_like(tmp_filename, solver_)
    t1 = time.perf_counter()
    b = solver_.simplify()
    t2 = time.perf_counter()
    result = solver_.solve_()
    t3 = time.perf_counter()

    key_stats = {
        'restarts': solver_.starts,
        'conflicts': solver_.conflicts,
        'decisions': solver_.decisions,
        'propagations': solver_.propagations_bcp,
        'conflict_literals': solver_.tot_literals,
        'time_ms': {
            'parse': (t1 - t0) * 1000.0,
            'simplify': (t2 - t1) * 1000.0,
            'solve': (t3 - t2) * 1000.0,
            'total': (t3 - t0) * 1000.0,
        },
    }
    return key_stats


def apply_imitsat_branch(file_name, cnf_str, imitsat_model_apply_fn, imitsat_tokenizer,
                         imitsat_param=None, context_len=0, latent_len=0, _argmax_last=None):
    """
    Run ImitSAT branching on one CNF and collect stats.

    Args:
    	file_name: Path to temporary DIMACS file.
    	cnf_str: CNF in DIMACS string form.
    	imitsat_model_apply_fn: ImitSAT apply function.
    	imitsat_tokenizer: ImitSAT tokenizer.
    	imitsat_param: ImitSAT parameters pytree.
    	context_len: Token context length.
    	latent_len: Latent length for cross‑attention.
    	_argmax_last: Argmax helper on logits.
    """
    imitsat_solver = ImitSATSolver()
    imitsat_solver.max_tokenizer_range = imitsat_tokenizer.total_vocab_size

    imitsat_solver.verbosity = 0
    imitsat_solver.context_len = context_len
    imitsat_solver.latent_len = latent_len
    imitsat_solver.max_imitsat_pick_num = 3

    cnf_str = cnf_str.splitlines()[1:]
    cnf_str = filter_lines_starting_with_nonzero_int(cnf_str)
    cnf_str = " ".join(cnf_str)
    cnf_str = cnf_str.replace("\n", " ")
    imitsat_solver.set_cnf(cnf_str)

    imitsat_solver.set_model_tokenizer(imitsat_model_apply_fn, imitsat_param, imitsat_tokenizer, _argmax_last)

    t0 = time.perf_counter()
    unsat = parse_dimacs_minisat_like(file_name, imitsat_solver)
    t1 = time.perf_counter()
    imitsat_solver.simplify()
    t2 = time.perf_counter()
    result = imitsat_solver.solve_()
    t3 = time.perf_counter()
    imitsat_stats = {
        'restarts': imitsat_solver.starts,
        'conflicts': imitsat_solver.conflicts,
        'decisions': imitsat_solver.decisions,
        'propagations': imitsat_solver.propagations_bcp,
        'conflict_literals': imitsat_solver.tot_literals,
        'num_imitsat_pick': imitsat_solver.num_imitsat_pick,
        'num_normal_pick': imitsat_solver.num_normal_pick,
        'time_ms': {
            'parse': (t1 - t0) * 1000.0,
            'simplify': (t2 - t1) * 1000.0,
            'solve': (t3 - t2) * 1000.0,  # = **walk time**, includes model
            'total': (t3 - t0) * 1000.0,
        },
        'model_time_ms': imitsat_solver.model_time_ms
    }
    return imitsat_stats


def parse_cnf_string_in_memory(cnf_str: str) -> CNF:
    return CNF(from_string=cnf_str)


def process_single_sat_problem(file_name: str, imitsat_model_apply_fn=None, imitsat_tokenizer=None, imitsat_param=None,
                               context_len=0, latent_len=0, _argmax_last=None) -> Dict:
    """
    Solve one CNF with baseline and ImitSAT to produce a stats record.

    Args:
    	file_name: Path to DIMACS file.
    	imitsat_model_apply_fn: ImitSAT apply function.
    	imitsat_tokenizer: ImitSAT tokenizer.
    	imitsat_param: ImitSAT parameters pytree.
    	context_len: Token context length.
    	latent_len: Latent length for cross‑attention.
    	_argmax_last: Argmax helper on logits.
    """
    cnf_formula = CNF()
    cnf_formula.from_file(file_name)

    minisat_solver = MinisatSolver()
    minisat_solver.verbosity = 0
    t0 = time.perf_counter()
    unsat = parse_dimacs_minisat_like(file_name, minisat_solver)
    t1 = time.perf_counter()
    minisat_solver.simplify()
    t2 = time.perf_counter()
    result = minisat_solver.solve_()
    t3 = time.perf_counter()

    key_trace = convert_keytrace_to_str(minisat_solver.key_trace_events)
    key_stats = apply_key_trace(file_name, key_trace)

    imitsat_stats = apply_imitsat_branch(file_name, cnf_formula.to_dimacs(), imitsat_model_apply_fn, imitsat_tokenizer,
                                         imitsat_param, context_len, latent_len, _argmax_last)

    cnf_dimacs = cnf_formula.to_dimacs()
    stats = {
        'cnf': cnf_dimacs,
        'n_v': cnf_formula.nv,
        'n_c': len(cnf_formula.clauses),
        'satisfiable': result,
        'key_trace': key_trace,
        'raw_stats': {
            'restarts': minisat_solver.starts,
            'conflicts': minisat_solver.conflicts,
            'decisions': minisat_solver.decisions,
            'propagations': minisat_solver.propagations_bcp,
            'conflict_literals': minisat_solver.tot_literals,
            'time_ms': {
                'parse': (t1 - t0) * 1000.0,
                'simplify': (t2 - t1) * 1000.0,
                'solve': (t3 - t2) * 1000.0,
                'total': (t3 - t0) * 1000.0,
            }
        },
        'key_stats': key_stats,
        'imitsat_stats': imitsat_stats
    }

    return stats


def _cast_matmuls_to_bf16(x):
    if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32 and x.ndim >= 2:
        return x.astype(jnp.bfloat16)
    return x


def warmup_runtime(params, imitsat_model_apply_fn, tokenizer, prefix_len, _argmax_last):
    x_host = np.zeros((1, prefix_len), dtype=np.int32)
    x_dev = x_host
    _ = jax.block_until_ready(x_dev)

    logits = imitsat_model_apply_fn(params, jax.random.PRNGKey(0), x_dev, 1, False)
    logits = jax.block_until_ready(logits)

    last_pos = prefix_len - 1
    tok_dev = _argmax_last(logits)
    _ = jax.block_until_ready(tok_dev)

    tok_dev[0].tolist()
    decoded_list = tokenizer.decode(tok_dev[0].tolist(), skip_special_tokens=False)


def _build_argmax_last():
    def _argmax_last(logits):
        predicted_tokens = jnp.argmax(logits, axis=-1)[0]
        return predicted_tokens

    return jax.jit(_argmax_last, static_argnums=())


def load_imitsat_model_for_inference(model_dir: str,
                                     model_config: str):
    """
    Load ImitSAT tokenizer, params, JITed apply, and helpers.

    Args:
    	model_dir: Directory containing tokenizer/params.
    	model_config: Path to JSON config file.
    """
    tokenizer = ImitSATTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))
    npz = np.load(os.path.join(model_dir, "ImitSAT.npz"), allow_pickle=True)

    flat_dict = {}
    for top_key in npz.files:
        sub_dict_array = npz[top_key]
        subdict = sub_dict_array[()]
        flat_dict[top_key] = subdict
    params = hk.data_structures.to_haiku_dict(flat_dict)
    params = jax.tree.map(_cast_matmuls_to_bf16, params)

    use_gpu = True
    if use_gpu:
        print(f"Using GPU.")
        params = jax.device_put(params)

    with open(model_config, "r") as f:
        config = json.load(f)

    context_len = config["context_len"]
    latent_len = config["latent_len"]
    max_context_len = config["max_context_len"]

    dropout_prob = config["dropout_prob"]
    num_transformers = config["num_transformers"]
    num_channels = config["num_channels"]
    num_heads = config["num_heads"]
    cross_attend_widening_factor = config["cross_attend_widening_factor"]
    transformer_widening_factor = config["transformer_widening_factor"]
    position_encoding_type = config["position_encoding_type"]

    vocab_size = len(tokenizer.vocab)

    def hk_forward_fn(input_ids, latent_len_, is_training=False):
        f = build_forward_fn(
            vocab_size, max_context_len, num_channels, num_heads, num_transformers,
            dropout_prob, cross_attend_widening_factor, transformer_widening_factor, position_encoding_type,
            tokenizer.pad_token_id)
        return f(input_ids, latent_len_, is_training)

    forward_transformed = hk.transform(hk_forward_fn)
    imitsat_model_apply_fn = forward_transformed.apply

    if use_gpu:
        imitsat_model_apply_fn = jax.jit(forward_transformed.apply, static_argnums=(3, 4))

    dummy = jnp.zeros((1, context_len), dtype=jnp.int32)
    out = imitsat_model_apply_fn(params, jax.random.PRNGKey(0), dummy, 1, False)
    out = jax.block_until_ready(out)

    _argmax_last = _build_argmax_last()
    warmup_runtime(params, imitsat_model_apply_fn, tokenizer, context_len, _argmax_last)

    return params, imitsat_model_apply_fn, tokenizer, context_len, latent_len, _argmax_last


def process_sat_problems(problems: list, model_dir: str, model_config: str) -> list:
    """
    Processes a list of SAT problems, runs MiniSat on each, and extracts relevant data.

    Args:
        problems (list): A list of SAT problems in CNF format.

    Returns:
        list: A list of dictionaries containing the original row, CNF content, and trace.
    """
    results = []
    tmp_filename = './tmp_problem.cnf'

    imitsat_params, imitsat_model_apply_fn, imitsat_tokenizer, context_len, latent_len, _argmax_last \
        = load_imitsat_model_for_inference(model_dir, model_config)
    for index, problem_line in tqdm(enumerate(problems)):
        cnf_formula = cnf_line_2_CNF_class(problem_line)
        write_temp_cnf_file(cnf_formula, filename=tmp_filename)
        stats = process_single_sat_problem(tmp_filename, imitsat_model_apply_fn, imitsat_tokenizer, imitsat_params,
                                           context_len, latent_len, _argmax_last)

        results.append(stats)

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    return results


def process_txt_file(problems_file: str,
                     save_dir: str,
                     output_json_file_pre: str,
                     model_dir: str,
                     model_config: str
                     ) -> None:
    """
    Read problems from a TXT file, run ImitSAT, and save JSON.

    Args:
    	problems_file: Path to TXT file of CNF lines.
    	save_dir: Output directory for JSON.
    	output_json_file_pre: Prefix for output filename.
    	model_dir: ImitSAT model directory.
    	model_config: ImitSAT config path.
    """
    problems = read_sat_problems_lines(problems_file)
    file_name = Path(problems_file).name.split('.')[0]
    save_json = save_dir + f'{output_json_file_pre}_{file_name}.json'
    results = process_sat_problems(problems, model_dir, model_config)
    save_dicts_to_json(results, save_json)

    print(f'Results saved to {save_json}')


def process_single_folder(folder_name: str, imitsat_params=None, imitsat_model_apply_fn=None,
                          imitsat_tokenizer=None, context_len=0, latent_len=0, _argmax_last=None) -> list:
    """
    Process all .cnf files in a folder with ImitSAT and return stats list.

    Args:
    	folder_name: Path to a folder containing .cnf files.
    	imitsat_params: ImitSAT parameters pytree.
    	imitsat_model_apply_fn: ImitSAT apply function.
    	imitsat_tokenizer: ImitSAT tokenizer.
    	context_len: Token context length.
    	latent_len: Latent length for cross‑attention.
    	_argmax_last: Argmax helper on logits.
    """
    results = []
    for fname in tqdm(os.listdir(folder_name)):
        if fname.lower().endswith('.cnf'):
            stats = process_single_sat_problem(os.path.join(folder_name, fname), imitsat_model_apply_fn,
                                               imitsat_tokenizer,
                                               imitsat_params,
                                               context_len,
                                               latent_len, _argmax_last)
            results.append(stats)
    return results


def process_folders(folders: str, save_dir: str, output_json_file_pre: str,
                    model_dir: str, model_config: str):
    """
    Process each subfolder of CNF files and write one JSON per subfolder.

    Args:
    	folders: Parent directory containing subfolders of .cnf files.
    	save_dir: Directory to save JSON outputs.
    	output_json_file_pre: Prefix for JSON filenames.
    	model_dir: ImitSAT model directory.
    	model_config: ImitSAT config path.
    """
    imitsat_params, imitsat_model_apply_fn, imitsat_tokenizer, context_len, latent_len, _argmax_last \
        = load_imitsat_model_for_inference(model_dir, model_config)
    for entry in os.scandir(folders):
        if entry.is_dir():
            subfolder_path = entry.name
            print(f"Processing subfolder: {subfolder_path}")

            results = process_single_folder(entry.path, imitsat_params=imitsat_params,
                                            imitsat_tokenizer=imitsat_tokenizer,
                                            imitsat_model_apply_fn=imitsat_model_apply_fn,
                                            context_len=context_len,
                                            latent_len=latent_len,
                                            _argmax_last=_argmax_last)
            save_dicts_to_json(results, save_dir + f'{output_json_file_pre}_{subfolder_path}.json')


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path, if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(dir_path: str) -> None:
    """Create directory if needed."""
    os.makedirs(dir_path, exist_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ImitSAT over CNF instances from a TXT file or over folders of .cnf files."
    )
    p.add_argument(
        "--mode", choices=["txt", "folders"], required=True,
        help="txt: read problems from a text file; folders: process each subfolder containing .cnf files."
    )
    p.add_argument("--txt-file", type=str, default="./dataset/testset/sat_5_15_5000.txt",
                   help="Path to the input TXT file (required for --mode txt).")
    p.add_argument("--folder", type=str,
                   help="Path to the parent folder containing subfolders (required for --mode folders).")

    p.add_argument(
        "--save-path", type=str, default="./output/imitsat/",
        help="For --mode txt: JSON output file path. For --mode folders: directory to write per-subfolder JSONs."
    )
    p.add_argument("--output-prefix", type=str, default="ImitSAT", help="Prefix for JSON filenames in folders mode.")

    p.add_argument("--model-dir", type=str, default="./model_ckpt/",
                   help="Path to the trained ImitSAT model directory.")
    p.add_argument(
        "--model-config", type=str, default="./model_config/ImitSAT_config.json",
        help="Optional path to the model config JSON."
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    model_config = args.model_config
    print(args)
    if args.mode == "txt":
        if not args.txt_file:
            parser.error("--txt-file is required when --mode txt")
        ensure_parent_dir(args.save_path)
        process_txt_file(
            problems_file=args.txt_file,
            save_dir=args.save_path,
            output_json_file_pre=args.output_prefix,
            model_dir=args.model_dir,
            model_config=model_config
        )
    else:
        if not args.folder:
            parser.error("--folder is required when --mode folders")
        ensure_dir(args.save_path)
        process_folders(
            folders=args.folder,
            save_dir=args.save_path,
            output_json_file_pre=args.output_prefix,
            model_dir=args.model_dir,
            model_config=model_config
        )


if __name__ == '__main__':
    main()
