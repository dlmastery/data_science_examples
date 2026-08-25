# NanoLlama Tokenizer — Subword & Chat Template Tokenizer

import json
import re
from typing import List, Dict, Tuple, Optional

SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "<|user|>": 3,
    "<|assistant|>": 4,
    "<|system|>": 5
}

class NanoTokenizer:
    def __init__(self):
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.vocab = dict(SPECIAL_TOKENS)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._build_default_vocab()

    def _build_default_vocab(self):
        """Build high-frequency English subwords, letters, digits, and code tokens."""
        # 1. ASCII Characters
        for i in range(32, 127):
            ch = chr(i)
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inv_vocab[idx] = ch

        # 2. Frequent Subwords and Words for Chat, Stories, and Python Code
        common_words = [
            "\n", "  ", "    ", " ", "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "I", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
            "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
            "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
            "no", "just", "him", "know", "take", "people", "into", "year", "your", "good",
            "some", "could", "them", "see", "other", "than", "then", "now", "look", "only",
            "come", "its", "over", "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new", "want", "because", "any",
            "these", "give", "day", "most", "us", "is", "are", "was", "were", "been", "has",
            "had", "AI", "Python", "code", "def", "return", "import", "print", "class",
            "function", "story", "once", "upon", "forest", "little", "dragon", "magic",
            "hello", "hi", "assistant", "helpful", "friendly", "explain", "why", "solve",
            "problem", "question", "answer", "reasoning", "step", "because", "therefore",
            "algorithm", "neural", "network", "llama", "model", "prompt", "token", "attention",
            "true", "false", "None", "elif", "else", "for", "while", "in", "range", "len"
        ]

        for w in common_words:
            if w not in self.vocab:
                idx = len(self.vocab)
                self.vocab[w] = idx
                self.inv_vocab[idx] = w

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Greedy longest-matching subword tokenization."""
        tokens = []
        if add_bos:
            tokens.append(self.special_tokens["<|bos|>"])

        i = 0
        n = len(text)
        while i < n:
            # Check for special tokens first
            matched_special = False
            for st, st_id in self.special_tokens.items():
                if text.startswith(st, i):
                    tokens.append(st_id)
                    i += len(st)
                    matched_special = True
                    break
            if matched_special:
                continue

            # Greedy match longest subword in vocab
            matched = False
            for l in range(min(16, n - i), 0, -1):
                sub = text[i: i + l]
                if sub in self.vocab:
                    tokens.append(self.vocab[sub])
                    i += l
                    matched = True
                    break

            if not matched:
                # Fallback to single character or byte
                ch = text[i]
                tokens.append(self.vocab.get(ch, self.special_tokens["<|pad|>"]))
                i += 1

        if add_eos:
            tokens.append(self.special_tokens["<|eos|>"])

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into string."""
        chars = []
        for t in tokens:
            if skip_special_tokens and t in self.special_tokens.values():
                continue
            chars.append(self.inv_vocab.get(t, ""))
        return "".join(chars)

    def format_chat(self, user_msg: str, system_msg: str = "You are NanoLlama, a helpful, brilliant AI assistant.") -> str:
        """Format message into structured instruction chat template."""
        return f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

    def tokenize_with_metadata(self, text: str) -> List[Dict]:
        """Tokenize text and return list of tokens with subword string, ID, and color tag."""
        token_ids = self.encode(text)
        result = []
        for tid in token_ids:
            sub = self.inv_vocab.get(tid, "")
            is_special = tid in self.special_tokens.values()
            result.append({
                "id": tid,
                "text": sub,
                "is_special": is_special,
                "length": len(sub)
            })
        return result

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab": self.vocab,
                "special_tokens": self.special_tokens
            }, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.special_tokens = data["special_tokens"]
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
