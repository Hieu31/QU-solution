from __future__ import annotations

import time
from pathlib import Path

from reparos.dependencies import require
from reparos.model.checkpoint import load_checkpoint
from reparos.model.decoding import beam_search
from reparos.tokenization import SentencePieceTokenizer


class ReferencePredictor:
    def __init__(self, checkpoint: str | Path, tokenizer: str | Path, device: str = 'cpu') -> None:
        self.torch = require('torch', 'reparos-train')
        self.tokenizer = SentencePieceTokenizer(tokenizer)
        self.model, self.state = load_checkpoint(checkpoint, tokenizer, device)
        self.device = self.torch.device(device)

    def predict(self, text: str, beam_size: int = 10) -> dict[str, object]:
        started = time.perf_counter()
        ids = self.tokenizer.encode(text)[: self.model.config.max_length]
        source = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        with self.torch.no_grad():
            beams = beam_search(self.model, source, beam_size=beam_size)
        hypotheses = [self.tokenizer.decode(tokens) for tokens, _ in beams]
        return {
            'input_query': text,
            'top1_query': hypotheses[0],
            'hypotheses': hypotheses,
            'sequence_scores': [score for _, score in beams],
            'changed': hypotheses[0] != text,
            'latency_ms': (time.perf_counter() - started) * 1000,
            'model_stage': self.state['stage'],
            'backend': 'reference',
        }
