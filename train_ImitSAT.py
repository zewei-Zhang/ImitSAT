# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
Train ImitSAT.

A pipeline to train an ImitSAT model (Perceiver-AR) with JAX/Haiku for CDCL branching.
This file includes building a model, loading ImitSAT datasets, training with pmap,
validating, and saving  to npz checkpoints
"""
import os
import json
import re

import argparse
from pathlib import Path
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import jax
import jax.numpy as jnp
import haiku as hk
import optax

import numpy as np
from perceiver_ar.perceiver_ar_model import PerceiverAR
from perceiver_ar.losses import compute_weighted_cross_entropy
from ImitSAT.ImiSAT_tokenizer import ImitSATTokenizer
from ImitSAT.ImitSAT_dataset import ImitSATDataset


def load_bucket_file(path: str | Path,
                     tokenizer,
                     context_len: int,
                     max_vid: int,
                     latent_len: int,
                     permute_vars: bool = True) -> Dataset:
    """
    Create an `ImitSATDataset` for one .jsonl.gz bucket.

    Args:
        path: Path to the bucket file.
        tokenizer: ImitSAT tokenizer.
        context_len: Max sequence length.
        max_vid: Max variable id in vocab.
        latent_len: Perceiver-AR latent length.
        permute_vars: Whether to permute variable ids.

    Returns:
        A `Dataset` for this bucket.
    """
    return ImitSATDataset(
        file_path=str(path),
        tokenizer=tokenizer,
        context_len=context_len,
        permute_vars=permute_vars,
        max_vid=max_vid,
        latent_len=latent_len,
    )


def extract_integers_file_name(p: Path):
    """
    Extract integers from a filename stem for numeric sorting.

    Args:
        p: File path.

    Returns:
        Tuple of ints found in the stem.
    """
    return tuple(int(x) for x in re.findall(r"\d+", p.stem))


def build_forward_fn(vocab_size, max_seq_len, num_channels, num_heads,
                     num_transformers, dropout_prob, cross_attend_widening_factor, transformer_widening_factor,
                     position_encoding_type, pad_id):
    """
    Build a Perceiver-AR forward function.

    Returns:
        Callable: (input_ids, latent_len, is_training) -> logits [B, S, V].
    """

    def forward_fn(input_ids, latent_len, is_training=True):
        """
        Perceiver-AR forward pass.

        Args:
            input_ids: int32 [B, S].
            latent_len: int scalar.
            is_training: Enable dropout if True.

        Returns:
            Logits float32 [B, S, vocab_size].
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
            use_remat=True
        )
        return out.input_events_logits

    return forward_fn


@partial(jax.pmap, axis_name='data', static_broadcasted_argnums=(4, 5, 6))
def pmapped_train_step(params_repl, rng, opt_state, batch, apply_fn, optimizer, latent_len):
    """
    One pmap training step (loss, grad, update) across replicas.

    Args:
        params_repl: Replicated params pytree.
        rng: Per-replica RNG key(s).
        opt_state: Replicated optimizer state.
        batch: Dict with "input_ids", "labels" [R, local_bs, S].
        apply_fn: Haiku apply function.
        optimizer: Optax optimizer.
        latent_len: Perceiver-AR latent length.

    Returns:
        (new_params_repl, new_opt_state_repl, loss_val_repl)
    """

    def loss_fn(p):
        logits_ = apply_fn(p, rng, batch["input_ids"], latent_len, is_training=True)

        loss, z_loss, weight_sum = compute_weighted_cross_entropy(
            logits=logits_,
            targets=batch["labels"],
            z_loss=1e-4,
            label_smoothing=0
        )
        del weight_sum

        total_loss = jnp.sum(loss, axis=1)
        total_loss = jnp.mean(total_loss)
        return total_loss, logits_

    (loss_val, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_repl)

    grads = jax.lax.pmean(grads, axis_name='data')
    loss_val = jax.lax.pmean(loss_val, axis_name='data')

    updates, new_opt_state = optimizer.update(grads, opt_state, params_repl)
    new_params = optax.apply_updates(params_repl, updates)
    return new_params, new_opt_state, loss_val


def partial_validation(params, eval_step, val_loader, latent_len, tokenizer, max_val_steps=4):
    """
    Quick validation over a few batches; returns (loss, token-acc).

    Args:
        params: Non-replicated params.
        eval_step: Callable(params, rng, inputs)->logits.
        val_loader: DataLoader for validation.
        latent_len: Latent length (unused in eval_step here).
        tokenizer: Tokenizer for sample prints.
        max_val_steps: Max number of batches to evaluate.

    Returns:
        (mean_loss: float, accuracy: float).
    """
    total_loss = 0.0
    steps = 0
    total_correct, total_tokens = 0, 0
    printed_samples = False
    rng = jax.random.PRNGKey(9999)
    for i, batch_dict in enumerate(val_loader):
        if i >= max_val_steps:
            break

        # forward pass
        b_input_ids = jnp.array(batch_dict["input_ids"].numpy(), dtype=jnp.int32)
        b_labels = jnp.array(batch_dict["labels"].numpy(), dtype=jnp.int32)

        rng, subkey = jax.random.split(rng)
        logits = eval_step(params, subkey, b_input_ids)

        loss, z_loss, weight_sum = compute_weighted_cross_entropy(
            logits=logits,
            targets=b_labels,
            z_loss=1e-4,
            label_smoothing=0
        )
        del weight_sum
        loss_val = jnp.sum(loss, axis=1)
        loss_val = jnp.mean(loss_val)

        total_loss += float(loss_val)
        steps += 1

        preds = jnp.argmax(logits, axis=-1)
        mask = (b_labels != -100)
        total_tokens += int(mask.sum())
        total_correct += int(((preds == b_labels) & mask).sum())

        if not printed_samples:
            printed_samples = True
            predicted_tokens = jnp.argmax(logits, axis=-1)
            predicted_tokens = np.array(predicted_tokens)

            gold_tokens = np.array(b_labels)
            sample_preds = min(2, predicted_tokens.shape[0])

            for b in range(sample_preds):
                pred_list = predicted_tokens[b].tolist()
                label_list = gold_tokens[b].tolist()

                pred_strs = []
                gold_strs = []
                for ptok, gtok in zip(pred_list, label_list):
                    if gtok == -100:
                        continue
                    p_txt = tokenizer.decode([ptok], skip_special_tokens=False)
                    g_txt = tokenizer.decode([gtok], skip_special_tokens=False)
                    pred_strs.append(p_txt)
                    gold_strs.append(g_txt)

                print(f"[Val sample] batch={i}, item={b}")
                print(f"  Pred: {pred_strs}")
                print(f"  Gold: {gold_strs}")
                print("")
    if steps == 0:
        return 0.0
    mean_loss = total_loss / steps
    accuracy = total_correct / max(1, total_tokens)
    return mean_loss, accuracy


def _get_lr_schedule(optimizer_config):
    """
    Linear warmup, then constant LR.

    Args:
        optimizer_config: Dict with keys base_lr, warmup_steps, warmup_initial_lr.
    """
    base_lr = optimizer_config["base_lr"]
    warmup_steps = optimizer_config["warmup_steps"]  # e.g. 10000
    init_lr = optimizer_config["warmup_initial_lr"]  # often 0.0

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    warmup_schedule = optax.linear_schedule(
        init_value=init_lr,
        end_value=base_lr,
        transition_steps=warmup_steps
    )

    constant_schedule = optax.constant_schedule(base_lr)

    schedule_fn = optax.join_schedules(
        schedules=[warmup_schedule, constant_schedule],
        boundaries=[warmup_steps]
    )
    return schedule_fn


def create_optimizer(optimizer_config):
    """
    Creates an Optax optimizer chain that:
      - uses Adam with b1=0.1, b2=0.999, eps=1e-8
      - uses a linear warmup => constant LR schedule
      - optionally clips gradients by global norm if max_norm>0

    Expects optimizer_config dict like:
      {
         "training_steps": 200000,
         "base_lr": 3e-4,
         "warmup_steps": 10000,
         "warmup_initial_lr": 0.0,
         "max_norm": 1.0
      }
    """
    schedule_fn = _get_lr_schedule(optimizer_config)

    optax_chain = []
    optax_chain.append(optax.scale_by_adam(
        b1=0.9, b2=0.999, eps=1e-8
    ))
    optax_chain.append(optax.scale_by_schedule(schedule_fn))
    optax_chain.append(optax.scale(-1))
    base_opt = optax.chain(*optax_chain)

    max_norm = optimizer_config.get("max_norm", 1.0)
    if max_norm > 0:
        final_opt = optax.chain(
            optax.clip_by_global_norm(max_norm),
            base_opt
        )
    else:
        final_opt = base_opt

    return final_opt


def _load_npz_params(npz_path: str) -> hk.Params:
    """
    Load pretrained model params from `npz_path/ImitSAT.npz`.

    Args:
        npz_path: Directory containing the npz.

    Returns:
        Haiku params pytree.
    """
    npz = np.load(os.path.join(npz_path, "ImitSAT.npz"), allow_pickle=True)
    flat_dict = {}

    for top_key in npz.files:
        sub_dict_array = npz[top_key]
        subdict = sub_dict_array[()]
        flat_dict[top_key] = subdict
    params = hk.data_structures.to_haiku_dict(flat_dict)
    return params


def main():
    """
    Trian ImitSAT in stages.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='../model_config/ImitSAT_config.json',
                        help="Path to the JSON config file.")
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="The folder contains trained model and tokenizer."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    batch_size = config["batch_size"]
    context_len = config["context_len"]
    max_context_len = config["max_context_len"]
    latent_len = config["latent_len"]

    num_epochs = config["num_epochs"]
    dropout_prob = config["dropout_prob"]
    num_transformers = config["num_transformers"]
    num_channels = config["num_channels"]
    num_heads = config["num_heads"]
    cross_attend_widening_factor = config["cross_attend_widening_factor"]
    transformer_widening_factor = config["transformer_widening_factor"]
    position_encoding_type = config["position_encoding_type"]

    optimizer_config = config["optimizer"]
    vocab_variable_range = config["vocab_variable_range"]
    dataset_path = config["dataset_path"]
    log_dir = config["tensorboard_log_dir"]
    out_dir = config["output_dir"]
    seed = config["seed"]
    val_every = config["val_every"]
    max_val_steps = config["max_val_steps"]
    save_every = config["save_every"]
    writer = SummaryWriter(log_dir=log_dir)

    if args.resume_dir:
        tokenizer = ImitSATTokenizer.from_pretrained(os.path.join(args.resume_dir, "tokenizer"))
        tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    else:
        sample_vocab = (
                [str(i) for i in range(vocab_variable_range + 1)]
                + [f"-{i}" for i in range(1, vocab_variable_range + 1)]
                + ["[CNF]", "[SEP]", "D", "[EOS]", "[UNK]"]
        )
        tokenizer = ImitSATTokenizer(sample_vocab)
        tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))

    current_file = config.get("current_file", None)  # e.g., "sat_5_15_jsonl.gz"
    replay_fraction = float(config.get("replay_fraction", 0.0))

    bucket_files = sorted(Path(dataset_path).glob("*.jsonl.gz"), key=extract_integers_file_name)
    print(f"All training datasets: {bucket_files}.")

    train_parts, val_parts, val_loaders = [], [], {}

    for p in bucket_files:
        ds_full = ImitSATDataset(
            str(p), tokenizer,
            context_len=context_len,
            permute_vars=True,
            max_vid=vocab_variable_range,
            latent_len=latent_len,
        )
        if p.name != current_file:
            keep_num = int(replay_fraction * len(ds_full))
            ds_full, _ = random_split(ds_full, [keep_num, len(ds_full) - keep_num])
            print(f"dataset: {p.name}, items: {len(ds_full)}")
        # 90 / 10 split inside *this* bucket
        n_val = max(1, int(0.1 * len(ds_full)))
        val_ds, train_ds = random_split(ds_full, [n_val, len(ds_full) - n_val])

        train_parts.append(train_ds)
        val_parts.append(val_ds)
        val_loaders[p.stem] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )

    train_ds = ConcatDataset(train_parts)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    global_val_loader = DataLoader(
        ConcatDataset(val_parts),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    print(f"Total {len(train_loader)} steps per epoch.")
    vocab_size = len(tokenizer.vocab)

    def hk_forward_fn(input_ids, latent_len_, is_training=True):
        f = build_forward_fn(
            vocab_size, max_context_len, num_channels, num_heads, num_transformers,
            dropout_prob, cross_attend_widening_factor, transformer_widening_factor, position_encoding_type,
            tokenizer.pad_token_id)
        return f(input_ids, latent_len_, is_training)

    forward_transformed = hk.transform(hk_forward_fn)

    eval_step = jax.jit(lambda params, rng, inputs:
                        forward_transformed.apply(
                            params,
                            rng,
                            inputs,
                            latent_len,
                            False
                        )
                        )

    rng = jax.random.PRNGKey(seed)
    if args.resume_dir:
        print(f"[Resume] Loading params from {args.resume_dir}")
        params = _load_npz_params(args.resume_dir)
    else:
        dummy_batch = next(iter(train_loader))
        db_input_ids = jnp.array(dummy_batch["input_ids"].numpy(), dtype=jnp.int32)
        db_latent_len = jnp.array(latent_len, dtype=jnp.int32)
        params = forward_transformed.init(rng, db_input_ids, db_latent_len, True)

    optimizer = create_optimizer(optimizer_config)
    opt_state = optimizer.init(params)

    device_count = jax.device_count()
    print(f"Found {device_count} devices.")
    params_repl = jax.tree.map(lambda x: jnp.stack([x] * device_count), params)
    opt_state_repl = jax.tree.map(lambda x: jnp.stack([x] * device_count), opt_state)
    rngs_repl = jax.tree.map(lambda x: jnp.stack([x] * device_count), rng)
    global_step = 0
    print(f"Total {len(train_loader)} steps per epoch.")

    for epoch in range(num_epochs):
        for step, batch_dict in enumerate(train_loader):
            global_step += 1
            input_ids_ = batch_dict["input_ids"].numpy()
            labels_ = batch_dict["labels"].numpy()

            gbs = input_ids_.shape[0]
            local_bs = gbs // device_count

            input_ids_ = input_ids_.reshape(device_count, local_bs, -1)
            labels_ = labels_.reshape(device_count, local_bs, -1)

            input_ids_jnp = jnp.array(input_ids_, dtype=jnp.int32)
            labels_jnp = jnp.array(labels_, dtype=jnp.int32)

            rngs_repl = jax.random.split(rngs_repl[0], device_count)

            params_repl, opt_state_repl, loss_val_repl = pmapped_train_step(
                params_repl, rngs_repl, opt_state_repl,
                {"input_ids": input_ids_jnp, "labels": labels_jnp},
                forward_transformed.apply,
                optimizer,
                latent_len
            )
            loss_val = float(loss_val_repl[0])

            if step % 10 == 0:
                writer.add_scalar("train/epoch", epoch, global_step)
                writer.add_scalar("train/loss", loss_val, global_step)

            if step % 100 == 0:
                print(f"Epoch {epoch} step {step}, loss={loss_val:.4f}")
            if step % 1000 == 0:
                writer.flush()

            if global_step % val_every == 0:
                single_device_params = jax.tree.map(lambda x: x[0], params_repl)
                glob_loss, glob_acc = partial_validation(
                    single_device_params, eval_step, global_val_loader,
                    latent_len, tokenizer, max_val_steps
                )
                writer.add_scalar("val/global_loss", glob_loss, global_step)
                writer.add_scalar("val/global_acc", glob_acc, global_step)

                for name, v_loader in val_loaders.items():
                    loss_b, acc_b = partial_validation(
                        single_device_params, eval_step, v_loader,
                        latent_len, tokenizer, max_val_steps
                    )
                    writer.add_scalar(f"val/{name}_loss", loss_b, global_step)
                    writer.add_scalar(f"val/{name}_acc", acc_b, global_step)

            if global_step % save_every == 0:
                final_params = jax.tree.map(lambda x: x[0, ...], params_repl)
                final_params_dict = hk.data_structures.to_mutable_dict(final_params)
                os.makedirs(out_dir, exist_ok=True)
                np.savez(os.path.join(out_dir, f"ImitSAT_step{global_step}.npz"), **final_params_dict)
                print(f"Done training. Check ImitSAT_step{global_step}.npz.")

        if (epoch + 1) % 1 == 0:
            final_params = jax.tree.map(lambda x: x[0, ...], params_repl)
            final_params_dict = hk.data_structures.to_mutable_dict(final_params)
            os.makedirs(out_dir, exist_ok=True)
            np.savez(os.path.join(out_dir, f"ImitSAT_epoch{epoch}.npz"), **final_params_dict)
            print(f"Done epoch training. Check ImitSAT_epoch{epoch}.npz.")

    writer.close()


if __name__ == "__main__":
    main()
