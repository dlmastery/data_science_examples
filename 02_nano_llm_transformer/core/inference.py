# NanoLlama High-Performance Inference Engine with Streaming, KV-Cache & Attention Extraction

import os
import sys
import json
import time
from typing import Generator, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from model import NanoLlama
from tokenizer import NanoTokenizer

CHECKPOINTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../server/checkpoints'))

class NanoInferenceEngine:
    def __init__(self, checkpoints_dir: str = CHECKPOINTS_DIR):
        self.checkpoints_dir = checkpoints_dir
        self.model: Optional[NanoLlama] = None
        self.tokenizer: Optional[NanoTokenizer] = None
        self.telemetry: Optional[Dict] = None
        self.load()

    def load(self):
        tokenizer_path = os.path.join(self.checkpoints_dir, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            self.tokenizer = NanoTokenizer()
            self.tokenizer.load(tokenizer_path)
            print(f"[OK] Loaded Tokenizer ({len(self.tokenizer.vocab)} tokens)")
        else:
            print("[WARN] Tokenizer file not found, initializing base tokenizer.")
            self.tokenizer = NanoTokenizer()

        model_path = os.path.join(self.checkpoints_dir, 'model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            cfg = checkpoint["config"]
            self.model = NanoLlama(**cfg)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            print(f"[OK] Loaded NanoLlama Model ({self.model.count_parameters():,} params)")
        else:
            print("[WARN] Model checkpoint not found.")

        meta_path = os.path.join(self.checkpoints_dir, 'telemetry.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.telemetry = json.load(f)

    @torch.no_grad()
    def generate_stream(
        self,
        user_message: str,
        system_prompt: str = "You are NanoLlama, a helpful, brilliant AI assistant.",
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1
    ) -> Generator[Dict, None, None]:
        """Streaming token-by-token generation with KV-Cache and live telemetry."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or Tokenizer not loaded")

        prompt_str = self.tokenizer.format_chat(user_message, system_prompt)
        prompt_tokens = self.tokenizer.encode(prompt_str)

        device = next(self.model.parameters()).device
        generated = list(prompt_tokens)
        eos_id = self.tokenizer.special_tokens["<|eos|>"]

        # Clear KV caches
        for layer in self.model.layers:
            layer.attention.cache_k = None
            layer.attention.cache_v = None

        t_start = time.time()
        t_first_token = None

        # Prefill prompt tokens into KV Cache
        tokens_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        logits, _, _ = self.model.forward(tokens_tensor, start_pos=0, use_cache=True)
        cur_pos = len(prompt_tokens)

        token_count = 0
        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            # Temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                # Top-P
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if t_first_token is None:
                t_first_token = time.time()

            generated.append(next_token)
            token_count += 1

            if next_token == eos_id:
                break

            subword = self.tokenizer.decode([next_token])
            elapsed = time.time() - t_start
            tok_sec = token_count / max(0.001, elapsed)

            yield {
                "token_id": next_token,
                "text": subword,
                "tokens_generated": token_count,
                "tokens_per_sec": round(tok_sec, 1),
                "time_to_first_token_ms": round((t_first_token - t_start) * 1000.0, 1) if t_first_token else 0.0,
                "is_finished": False
            }

            if cur_pos >= self.model.max_seq_len - 1:
                break

            # Next token inference step (KV Cache enabled)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, _, _ = self.model.forward(next_token_tensor, start_pos=cur_pos, use_cache=True)
            cur_pos += 1

        total_elapsed = time.time() - t_start
        yield {
            "token_id": eos_id,
            "text": "",
            "tokens_generated": token_count,
            "tokens_per_sec": round(token_count / max(0.001, total_elapsed), 1),
            "total_time_sec": round(total_elapsed, 2),
            "is_finished": True
        }

    @torch.no_grad()
    def get_attention_matrix(self, prompt: str) -> Dict:
        """Extract multi-head attention weights for all layers for frontend heatmap visualization."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or Tokenizer not loaded")

        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > 48:
            tokens = tokens[:48]  # Bounded for clean heatmap visualization

        tokens_tensor = torch.tensor([tokens], dtype=torch.long)
        _, _, attentions = self.model.forward(tokens_tensor, return_attentions=True)

        token_labels = [self.tokenizer.decode([t]) for t in tokens]

        layers_data = []
        for l_idx, layer_attn in enumerate(attentions):
            # layer_attn shape: (1, n_heads, seqlen, seqlen)
            heads_data = []
            for h_idx in range(self.model.n_heads):
                matrix = layer_attn[0, h_idx].tolist()
                heads_data.append({
                    "head_id": h_idx,
                    "matrix": matrix
                })
            layers_data.append({
                "layer_id": l_idx,
                "heads": heads_data
            })

        return {
            "tokens": token_labels,
            "token_ids": tokens,
            "layers": layers_data
        }

    @torch.no_grad()
    def inspect_tokens_and_probabilities(self, text: str) -> Dict:
        """Analyze subword tokenization and top-5 predicted next tokens."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or Tokenizer not loaded")

        token_metadata = self.tokenizer.tokenize_with_metadata(text)
        tokens = [t["id"] for t in token_metadata]

        tokens_tensor = torch.tensor([tokens], dtype=torch.long)
        logits, _, _ = self.model.forward(tokens_tensor)
        last_logits = logits[0, -1, :]

        probs = F.softmax(last_logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, 5)

        top_predictions = []
        for p, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
            top_predictions.append({
                "token_id": idx,
                "text": self.tokenizer.decode([idx]),
                "probability": round(p * 100.0, 2)
            })

        return {
            "tokens": token_metadata,
            "total_tokens": len(tokens),
            "top_predictions": top_predictions
        }

engine = NanoInferenceEngine()
