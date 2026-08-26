# NanoLlama Inference Engine — KV-Cache Autoregressive Generation & Dynamic Attention Telemetry

import os
import sys
import time
import json
from typing import Generator, Dict, Any, List, Optional
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
        self.tokenizer = NanoTokenizer()
        self.model: Optional[NanoLlama] = None
        self.config: Dict[str, Any] = {}
        self.telemetry: Optional[Dict[str, Any]] = None
        self.load = self.load_model
        self.load_model()

    def load_model(self):
        tok_path = os.path.join(self.checkpoints_dir, 'tokenizer.json')
        model_path = os.path.join(self.checkpoints_dir, 'model.pt')

        if os.path.exists(tok_path):
            self.tokenizer.load(tok_path)
            print(f"[OK] Loaded Tokenizer ({len(self.tokenizer.vocab)} tokens)")

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            self.config = ckpt.get("config", {})
            self.model = NanoLlama(**self.config)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            print(f"[OK] Loaded NanoLlama Model ({self.model.count_parameters():,} params)")
        else:
            print("[WARN] Checkpoint model.pt not found. Run train.py first.")

    @torch.no_grad()
    def generate_stream(
        self,
        user_message: str,
        system_prompt: str = "You are NanoLlama, a helpful AI assistant.",
        max_new_tokens: int = 250,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05
    ) -> Generator[Dict[str, Any], None, None]:
        if self.model is None:
            raise RuntimeError("Model is not initialized or checkpoints are missing.")

        # Normalize system prompt to clean canonical form if empty
        if not system_prompt or not system_prompt.strip():
            system_prompt = "You are NanoLlama, a helpful AI assistant."

        prompt_str = self.tokenizer.format_chat(user_message.strip(), system_prompt.strip())
        prompt_tokens = self.tokenizer.encode(prompt_str, add_bos=True)

        device = next(self.model.parameters()).device
        eos_id = self.tokenizer.special_tokens["<|eos|>"]

        generated_tokens = list(prompt_tokens)
        assistant_tokens = []
        t_start = time.time()
        t_first_token = None

        token_count = 0
        for _ in range(max_new_tokens):
            if len(generated_tokens) >= self.model.max_seq_len:
                break

            inp = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            logits, _, _ = self.model(inp)
            next_token_logits = logits[:, -1, :].clone()

            # Repetition penalty applied exclusively to assistant output tokens
            if repetition_penalty != 1.0 and len(assistant_tokens) > 0:
                for token_id in set(assistant_tokens):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            # Temperature and Top-K/P Sampling
            if temperature > 0.15:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

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
                # Deterministic argmax for maximum coherence with character-level models
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if t_first_token is None:
                t_first_token = time.time()

            if next_token == eos_id:
                break

            generated_tokens.append(next_token)
            assistant_tokens.append(next_token)
            token_count += 1

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
    def get_attention_matrix(self, prompt: str) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not loaded"}

        prompt_tokens = self.tokenizer.encode(prompt, add_bos=True)
        if len(prompt_tokens) > 64:
            prompt_tokens = prompt_tokens[:64]

        device = next(self.model.parameters()).device
        tokens_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        _, _, attn_matrices = self.model.forward(tokens_tensor, return_attentions=True)
        if not attn_matrices:
            return {"error": "No attention weights produced"}

        # Return attention for Layer 0, Head 0
        layer_0_attn = attn_matrices[0][0, 0].cpu().numpy().tolist()
        tokens_decoded = [self.tokenizer.decode([t]) for t in prompt_tokens]

        return {
            "tokens": tokens_decoded,
            "token_ids": prompt_tokens,
            "layer": 0,
            "head": 0,
            "attention_matrix": layer_0_attn
        }

    @torch.no_grad()
    def inspect_tokens_and_probabilities(self, text: str) -> Dict[str, Any]:
        tokens = self.tokenizer.encode(text, add_bos=True)
        tokens_decoded = [self.tokenizer.decode([t]) for t in tokens]
        
        top5 = []
        if self.model is not None and len(tokens) > 0:
            device = next(self.model.parameters()).device
            inp = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _, _ = self.model.forward(inp)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(5, len(probs)))
            for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                top5.append({
                    "token_id": idx,
                    "token_str": self.tokenizer.decode([idx]),
                    "probability": round(p, 4)
                })

        return {
            "tokens": tokens_decoded,
            "token_ids": tokens,
            "count": len(tokens),
            "top5_predictions": top5
        }

engine = NanoInferenceEngine()
