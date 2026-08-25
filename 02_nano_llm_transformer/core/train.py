# NanoLlama Training Pipeline — Cosine Annealing, AdamW, Perplexity & Checkpointing

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from model import NanoLlama
from tokenizer import NanoTokenizer
from dataset import ChatDataset

CHECKPOINTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../server/checkpoints'))

def train_nanollama(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-3,
    max_seq_len: int = 96,
    export_dir: str = CHECKPOINTS_DIR
):
    os.makedirs(export_dir, exist_ok=True)
    print("🚀 Initializing NanoLlama SOTA Model & Tokenizer Training...", flush=True)
    start_time = time.time()

    # 1. Tokenizer
    tokenizer = NanoTokenizer()
    vocab_size = len(tokenizer.vocab)
    print(f"📖 Tokenizer Vocabulary: {vocab_size} tokens", flush=True)

    # 2. Model Initialization
    dim = 128
    n_layers = 3
    n_heads = 4
    ffn_hidden_dim = 384

    model = NanoLlama(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        max_seq_len=max_seq_len
    )

    param_count = model.count_parameters()
    print(f"🧠 Model Architecture: {n_layers} Layers | {n_heads} Heads (dim {dim//n_heads}) | RoPE | SwiGLU ({ffn_hidden_dim}) | RMSNorm", flush=True)
    print(f"📊 Total Trainable Parameters: {param_count:,}", flush=True)

    # 3. Dataset & DataLoader
    dataset = ChatDataset(tokenizer=tokenizer, max_seq_len=max_seq_len, repeat_factor=15)
    train_size = int(len(dataset) * 0.85)
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"🧪 Training Samples: {len(train_ds)} | Validation Samples: {len(val_ds)}")

    # 4. Optimizer & Cosine LR Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = epochs * len(train_loader)
    warmup_steps = max(10, int(total_steps * 0.05))

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.1 + 0.5 * lr * 0.9 * (1.0 + math.cos(math.pi * progress))

    # 5. Training Loop
    telemetry_loss_curve = []
    telemetry_val_curve = []
    global_step = 0

    print("\n🏋️ Starting Optimization Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            # Update learning rate
            curr_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

            optimizer.zero_grad()
            _, loss, _ = model(batch_x, targets=batch_y)
            loss.backward()

            # Gradient Clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            if global_step % 5 == 0:
                telemetry_loss_curve.append({
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": round(loss_val, 4),
                    "learning_rate": round(curr_lr, 6),
                    "grad_norm": round(float(grad_norm), 3)
                })

            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                _, v_loss, _ = model(vx, targets=vy)
                val_loss += v_loss.item()
        avg_val_loss = val_loss / len(val_loader)
        perplexity = math.exp(min(20.0, avg_val_loss))

        telemetry_val_curve.append({
            "epoch": epoch,
            "val_loss": round(avg_val_loss, 4),
            "perplexity": round(perplexity, 2),
            "epoch_duration_sec": round(time.time() - t0, 2)
        })

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  ➔ Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f} | lr: {curr_lr:.5f}", flush=True)

    # 6. Save Checkpoints & Telemetry
    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, 'model.pt')
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "dim": dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ffn_hidden_dim": ffn_hidden_dim,
            "max_seq_len": max_seq_len
        }
    }, model_path)
    print(f"\n💾 Model Weights saved to: {model_path}", flush=True)

    tokenizer_path = os.path.join(export_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"💾 Tokenizer Vocab saved to: {tokenizer_path}", flush=True)

    metadata_path = os.path.join(export_dir, 'telemetry.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": "NanoLlama-SOTA",
            "architecture": "Decoder-Only Transformer (RoPE + SwiGLU + RMSNorm)",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "parameters_count": param_count,
            "config": {
                "vocab_size": vocab_size,
                "embedding_dim": dim,
                "num_layers": n_layers,
                "num_heads": n_heads,
                "head_dim": dim // n_heads,
                "ffn_hidden_dim": ffn_hidden_dim,
                "context_window": max_seq_len
            },
            "final_metrics": {
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "perplexity": round(perplexity, 2),
                "total_training_time_sec": round(time.time() - start_time, 2)
            },
            "training_curve": telemetry_loss_curve,
            "validation_curve": telemetry_val_curve
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"✨ Training complete in {total_time:.2f}s! Final Perplexity: {perplexity:.2f}", flush=True)

if __name__ == '__main__':
    train_nanollama()
