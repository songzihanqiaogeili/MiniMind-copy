"""单机预训练：从项目根目录运行 python -m trainer.train_pretrain --help。"""

import argparse
from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.llm_dataset import PretrainDataset
from model.model import MokioMindConfig, MokioMindForCausalLM
from trainer.train_utils import get_lr, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind single-device pretraining")
    parser.add_argument("--data_path", required=True, help="UTF-8 JSONL file with text fields")
    parser.add_argument("--tokenizer_path", required=True, help="Local tokenizer directory")
    parser.add_argument("--save_dir", default="out/pretrain")
    parser.add_argument("--from_model", help="Saved model directory; starts a new optimizer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--log_interval", type=int, default=10, help="Optimizer updates")
    parser.add_argument("--save_interval", type=int, default=100, help="Optimizer updates")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N optimizer updates; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "accumulation_steps", "log_interval", "save_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0 or args.max_steps < 0 or args.max_seq_len < 2:
        parser.error("Invalid worker count, max_steps or max_seq_len")
    if args.learning_rate <= 0 or args.grad_clip <= 0:
        parser.error("learning_rate and grad_clip must be positive")
    return args


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This trainer supports CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if device.type == "cpu" and args.dtype != "float32":
        raise ValueError("This trainer uses float32 on CPU; set --dtype float32")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.from_model:
        model = MokioMindForCausalLM.from_pretrained(args.from_model, local_files_only=True)
        if model.config.vocab_size != len(tokenizer):
            raise ValueError("Tokenizer vocabulary size does not match saved model")
        if args.max_seq_len > model.config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds saved model context length")
    else:
        config = MokioMindConfig(
            vocab_size=len(tokenizer), hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            max_position_embeddings=args.max_seq_len,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = MokioMindForCausalLM(config)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.dtype == "float16")
    dtype = getattr(torch, args.dtype)
    steps_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    save_dir = Path(args.save_dir)
    # 避免覆盖已有训练输出。
    save_dir.mkdir(parents=True, exist_ok=True)
    if any(save_dir.iterdir()):
        raise ValueError("save_dir must be empty; choose a new output directory")

    def save(name):
        destination = save_dir / name
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)

    print(f"device={device} samples={len(dataset)} parameters={sum(p.numel() for p in model.parameters()):,} updates={total_steps}")
    update = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        group_tokens = 0
        group_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            context = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and args.dtype != "float32" else nullcontext()
            with context:
                result = model(**batch)
                # 先累加 token loss 总和，更新前按有效 token 数归一化。
                # 这样末尾不足 accumulation_steps 的组也有正确权重。
                loss = (result.loss + result.aux_loss) * tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            group_tokens += tokens
            group_loss += loss.detach().float().item()
            if step % args.accumulation_steps != 0 and step != len(loader):
                continue
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(max(group_tokens, 1))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = get_lr(update, total_steps, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update == 1 or update % args.log_interval == 0 or update == total_steps:
                print(f"epoch={epoch+1} update={update}/{total_steps} loss={group_loss/max(group_tokens,1):.6f} lr={lr:.8g}", flush=True)
            group_tokens = 0
            group_loss = 0.0
            if update % args.save_interval == 0:
                save(f"step-{update}")
            if update >= total_steps:
                break
        if update >= total_steps:
            break
    save("final")
    print(f"Saved model and tokenizer to {save_dir / 'final'}")


if __name__ == "__main__":
    main()
