"""
Embedding 提取脚本

从 Evo2 模型中提取序列 embedding，支持基础模型和 LoRA 微调模型。
利用 BioNeMo 的 predict_evo2 基础设施，通过 forward hook 捕获中间层激活。

用法:
    # 使用基础模型
    python src/extract_embeddings.py \
        --fasta data/raw/igem_sequences.fasta \
        --ckpt-dir ~/.cache/bionemo/...-nemo2_evo2_1b_8k_bf16.tar.gz.untar \
        --output results/embeddings/embeddings.npz

    # 使用 LoRA 微调模型
    python src/extract_embeddings.py \
        --fasta data/raw/igem_sequences.fasta \
        --ckpt-dir ~/.cache/bionemo/...-nemo2_evo2_1b_8k_bf16.tar.gz.untar \
        --lora-checkpoint-path results/evo2_1b_lora_ft/checkpoints/epoch=0-step=4999-consumed_samples=5000.0-last \
        --output results/embeddings/embeddings.npz
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Evo2 embeddings from FASTA sequences")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file")
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="Base model checkpoint directory")
    parser.add_argument("--lora-checkpoint-path", type=Path, default=None,
                        help="Path to LoRA checkpoint (optional)")
    parser.add_argument("--output", type=Path, default=Path("results/embeddings/embeddings.npz"))
    parser.add_argument("--layer", type=int, default=16,
                        help="Which model layer to extract embeddings from (default: 16)")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean",
                        help="Pooling strategy: mean (recommended) or max")
    parser.add_argument("--model-size", type=str, default="1b",
                        help="Model size (default: 1b)")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Max sequence length; longer sequences are truncated (default: 4096)")
    return parser.parse_args()


def read_fasta(fasta_path: Path):
    """Simple FASTA parser. Yields (header, sequence) tuples."""
    header = None
    seq_lines = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            elif line:
                seq_lines.append(line.upper())
    if header is not None:
        yield header, "".join(seq_lines)


# Global list to accumulate embeddings from forward hook
_all_captured = []


def extract_embeddings(args):
    """Extract embeddings using predict_evo2's predict() with a forward hook on intermediate layers."""
    import functools

    import nemo.lightning as nl
    from lightning.pytorch import LightningDataModule
    from lightning.pytorch.callbacks import Callback
    from megatron.core import parallel_state
    from megatron.core.utils import get_batch_on_this_cp_rank
    from nemo.collections.llm.gpt.model.base import get_packed_seq_params
    from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    from nemo.lightning import NeMoLogger
    from nemo.lightning.data import WrappedDataLoader

    from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
    from bionemo.evo2.models.peft import Evo2LoRA
    from bionemo.llm.data import collate
    from bionemo.llm.lightning import LightningPassthroughPredictionMixin

    global _all_captured
    _all_captured = []
    target_layer = args.layer
    pooling = args.pooling

    # Forward & data step (same as predict_evo2)
    def fwd_step(model, batch):
        forward_args = {
            "input_ids": batch["tokens"],
            "position_ids": batch["position_ids"],
            "attention_mask": None,
        }
        if "cu_seqlens" in batch:
            forward_args["packed_seq_params"] = get_packed_seq_params(batch)
        return model(**forward_args)

    def data_step(dataloader_iter):
        batch = next(dataloader_iter)
        _batch = batch[0] if isinstance(batch, tuple) and len(batch) == 3 else batch
        required_device_keys = {"attention_mask"}
        if "cu_seqlens" in _batch:
            required_device_keys.add("cu_seqlens")
        if parallel_state.is_pipeline_first_stage():
            required_device_keys.update(("tokens", "position_ids"))
        if parallel_state.is_pipeline_last_stage():
            required_device_keys.update(("labels", "tokens", "loss_mask"))
        out = {}
        for key, val in _batch.items():
            if key in required_device_keys:
                out[key] = val.cuda(non_blocking=True)
            else:
                out[key] = None
        result = get_batch_on_this_cp_rank(out)
        if parallel_state.is_pipeline_last_stage():
            result["seq_idx"] = _batch["seq_idx"].cuda(non_blocking=True)
        return result

    # Predictor - use the standard LightningPassthroughPredictionMixin
    class EmbPredictor(LightningPassthroughPredictionMixin, HyenaModel):
        def configure_model(self, *a, **kw):
            super().configure_model(*a, **kw)
            self.trainer.strategy._init_model_parallel = True

        def predict_step(self, batch, batch_idx=None):
            if len(batch) == 0:
                return None
            with torch.no_grad():
                self.forward_step(batch)
            # Return dummy logits so Megatron pipeline doesn't complain
            return None

    # Callback to register hook after model is configured
    class EmbeddingHookCallback(Callback):
        def __init__(self, layer_idx, pool):
            super().__init__()
            self.layer_idx = layer_idx
            self.pool = pool
            self._handle = None

        def on_predict_start(self, trainer, pl_module):
            # Navigate through Megatron's DDP wrapping to find decoder layers
            model = pl_module
            # Try various paths to reach the decoder
            decoder = None
            for attr_chain in [
                lambda m: m.module.module.decoder,      # DDP -> MegatronModule -> decoder
                lambda m: m.module.decoder,              # MegatronModule -> decoder
                lambda m: m.decoder,                     # direct
            ]:
                try:
                    decoder = attr_chain(model)
                    break
                except (AttributeError, TypeError):
                    continue

            if decoder is None:
                # Walk the module tree to find decoder
                for name, mod in model.named_modules():
                    if name.endswith('.decoder') and hasattr(mod, 'layers'):
                        decoder = mod
                        break

            if decoder is None:
                raise RuntimeError("Could not find decoder in model")

            layers = decoder.layers
            idx = min(self.layer_idx, len(layers) - 1)
            print(f"Registering embedding hook on layer {idx} / {len(layers)}")

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # hidden: (seq_len, batch, dim) or (batch, seq_len, dim)
                h = hidden.detach().float()
                if h.dim() == 3:
                    # Ensure (batch, seq_len, dim)
                    if h.shape[0] < h.shape[1]:  # likely (batch, seq, dim) already
                        pass
                    else:  # (seq, batch, dim)
                        h = h.transpose(0, 1)
                    for i in range(h.shape[0]):
                        vec = h[i]
                        if self.pool == "mean":
                            emb = vec.mean(dim=0)
                        else:
                            emb = vec.max(dim=0).values
                        _all_captured.append(emb.cpu().numpy())
                elif h.dim() == 2:
                    if self.pool == "mean":
                        emb = h.float().mean(dim=0)
                    else:
                        emb = h.float().max(dim=0).values
                    _all_captured.append(emb.cpu().numpy())
                # Print progress
                n = len(_all_captured)
                if n % 100 == 0 or n <= 5:
                    print(f"\r  Processed {n} sequences...", end="", flush=True)

            self._handle = layers[idx].register_forward_hook(hook_fn)

        def on_predict_end(self, trainer, pl_module):
            if self._handle:
                self._handle.remove()

    # DataModule
    class EmbDataModule(LightningDataModule):
        def __init__(self, dataset, batch_size=1, max_length=4096):
            super().__init__()
            self.dataset = dataset
            self.batch_size = batch_size
            self.max_length = max_length

        def setup(self, stage=None):
            pass

        def predict_dataloader(self):
            max_len = self.max_length

            def truncate_collate_fn(batch):
                """Truncate sequences to max_length before padding."""
                for item in batch:
                    for key in ("tokens", "position_ids", "loss_mask"):
                        if key in item and len(item[key]) > max_len:
                            item[key] = item[key][:max_len]
                return collate.padding_collate_fn(
                    batch,
                    padding_values={"tokens": 0, "position_ids": 0, "loss_mask": False},
                    min_length=None,
                    max_length=None,
                )

            return WrappedDataLoader(
                mode="predict",
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=False,
                drop_last=False,
                collate_fn=truncate_collate_fn,
            )

    tokenizer = get_nmt_tokenizer("byte-level")
    config = HYENA_MODEL_OPTIONS[args.model_size](
        forward_step_fn=fwd_step,
        data_step_fn=data_step,
        distribute_saved_activations=True,
    )

    callbacks = [EmbeddingHookCallback(target_layer, pooling)]
    model_transform = None
    if args.lora_checkpoint_path:
        model_transform = Evo2LoRA(peft_ckpt_path=str(args.lora_checkpoint_path))
        callbacks.append(model_transform)

    model = EmbPredictor(
        config,
        tokenizer=tokenizer,
        model_transform=model_transform,
    )

    work_dir = Path(tempfile.mkdtemp())
    trainer = nl.Trainer(
        accelerator="gpu",
        num_nodes=1,
        devices=1,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=args.micro_batch_size,
                global_batch_size=args.micro_batch_size,
                seq_len=args.max_length,
                output_log=False,
            ),
        ),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )
    trainer.strategy._setup_optimizers = False

    nemo_logger = NeMoLogger(log_dir=work_dir)
    nemo_logger.setup(trainer, resume_if_exists=True)
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        resume_from_path=str(args.ckpt_dir),
        restore_config=None,
    )
    resume.setup(trainer, model)

    dataset = SimpleFastaDataset(args.fasta, tokenizer, prepend_bos=False)
    datamodule = EmbDataModule(dataset, batch_size=args.micro_batch_size, max_length=args.max_length)

    print(f"Extracting embeddings from layer {target_layer} with {pooling} pooling...")
    print(f"Processing {len(dataset)} sequences...")
    trainer.predict(model, datamodule=datamodule)

    embeddings = np.stack(_all_captured)
    headers = [h for h, _ in read_fasta(args.fasta)]

    print(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    return headers, embeddings


def main():
    args = parse_args()

    if not args.fasta.exists():
        print(f"ERROR: FASTA file not found: {args.fasta}")
        sys.exit(1)

    if not args.ckpt_dir.exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt_dir}")
        sys.exit(1)

    headers, embeddings = extract_embeddings(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        headers=np.array(headers, dtype=object),
    )
    print(f"Saved embeddings to {args.output}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  File size: {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
