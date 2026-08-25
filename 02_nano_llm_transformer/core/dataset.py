# NanoLlama Conversational Instruction & Story Dataset

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from tokenizer import NanoTokenizer

TOY_DIALOGUES = [
    # 1. AI Assistant & Core Knowledge
    {
        "user": "Hello! Who are you?",
        "assistant": "Hello! I am NanoLlama, a modern neural language model built from scratch with Rotary Position Embeddings (RoPE), SwiGLU activations, and RMSNorm. How can I help you today?"
    },
    {
        "user": "What is Rotary Position Embedding (RoPE)?",
        "assistant": "RoPE is a position embedding technique that applies a 2D rotation matrix to query and key vectors in complex space. This allows the model to capture relative distances naturally between tokens without fixed absolute position lookup tables."
    },
    {
        "user": "Why do modern LLMs use SwiGLU instead of standard ReLU?",
        "assistant": "SwiGLU multiplies a Swish-activated linear projection with an un-activated linear projection. This non-linear gating mechanism provides superior gradient flow, expressiveness, and faster convergence during training."
    },
    {
        "user": "Explain what RMSNorm does in a Transformer.",
        "assistant": "RMSNorm normalizes input activations by dividing them by the Root Mean Square of their values. Unlike LayerNorm, it skips the mean-centering step, making it computationally faster while maintaining identical training stability."
    },
    {
        "user": "How does a Key-Value (KV) Cache speed up text generation?",
        "assistant": "During autoregressive text generation, the KV Cache stores the key and value vectors of previous tokens. When generating the next token, the model only computes Q, K, V for the single new token, reducing generation time complexity from O(N^2) to O(N)."
    },

    # 2. Python Coding Snippets
    {
        "user": "Write a Python function to reverse a string.",
        "assistant": "Here is the Python function:\n\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n\n# Example usage:\nprint(reverse_string('llama'))  # Output: 'amall'"
    },
    {
        "user": "Write a Python function to compute the Fibonacci sequence.",
        "assistant": "Here is the Fibonacci function in Python:\n\ndef fibonacci(n: int) -> list:\n    if n <= 0:\n        return []\n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[-1] + fib[-2])\n    return fib[:n]"
    },
    {
        "user": "How do you calculate factorial in Python?",
        "assistant": "You can calculate factorial recursively or iteratively:\n\ndef factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # Output: 120"
    },
    {
        "user": "Write a Python function to check if a number is prime.",
        "assistant": "Here is a clean primality test in Python:\n\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    },

    # 3. Creative Mini Stories (TinyStories Style)
    {
        "user": "Tell me a short bedtime story about a baby dragon.",
        "assistant": "Once upon a time in a magical forest, there lived a little dragon named Spark. Spark loved to light tiny glowing embers to help the lost fireflies find their way home. Every night, the forest animals gathered around Spark to hear tales of the stars, drifting peacefully to sleep under the warm moonlight."
    },
    {
        "user": "Write a story about a brave astronaut exploring Mars.",
        "assistant": "Astronaut Maya stepped out onto the red soil of Mars. As the dust settled, she discovered a crystalline cave glowing with azure light. Inside, ancient frozen water sparkled like diamonds. Maya smiled and radioed Earth: 'We are not alone; the planet has stories to tell.'"
    },
    {
        "user": "Write a story about a wise owl and a curious squirrel.",
        "assistant": "High atop an oak tree, Barnaby the owl watched Pippin the squirrel frantically hoard acorns. 'Why do you rush so, little one?' Barnaby asked. Pippin replied, 'Winter is coming!' The owl smiled and said, 'Prepare well, but remember to enjoy the autumn breeze today.'"
    },

    # 4. Math & Logic Reasoning
    {
        "user": "If a train travels 120 miles in 2 hours, what is its average speed?",
        "assistant": "To find the average speed, divide distance by time:\nSpeed = Distance / Time\nSpeed = 120 miles / 2 hours = 60 miles per hour."
    },
    {
        "user": "What is 15% of 80?",
        "assistant": "To calculate 15% of 80:\n1. 10% of 80 is 8.\n2. 5% of 80 is 4 (half of 8).\n3. 15% = 8 + 4 = 12.\nTherefore, 15% of 80 is 12."
    },
    {
        "user": "Solve for x: 3x + 9 = 24",
        "assistant": "Step 1: Subtract 9 from both sides: 3x = 24 - 9 = 15.\nStep 2: Divide both sides by 3: x = 15 / 3 = 5.\nTherefore, x = 5."
    }
]

class ChatDataset(Dataset):
    """Padded PyTorch Dataset for causal autoregressive instruction training."""
    def __init__(self, tokenizer: NanoTokenizer, max_seq_len: int = 192, repeat_factor: int = 15):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        for _ in range(repeat_factor):
            for d in TOY_DIALOGUES:
                text = (
                    f"<|system|>\nYou are NanoLlama, a helpful, brilliant AI assistant.\n"
                    f"<|user|>\n{d['user']}\n"
                    f"<|assistant|>\n{d['assistant']}<|eos|>"
                )
                tokens = tokenizer.encode(text)
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                self.samples.append(tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.samples[idx]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            target_ids = target_ids + [0] * pad_len
        else:
            input_ids = input_ids[:self.max_seq_len]
            target_ids = target_ids[:self.max_seq_len]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )
