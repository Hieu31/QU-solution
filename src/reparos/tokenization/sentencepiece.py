from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from reparos.config import TokenizerConfig
from reparos.dependencies import require
from reparos.manifests import fingerprint_files, sha256_file, write_json


def train_sentencepiece(
    inputs: Iterable[str | Path], output: str | Path,
    config: TokenizerConfig | None = None,
) -> dict[str, object]:
    settings = config or TokenizerConfig()
    settings.validate()
    files = [Path(item) for item in inputs]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError('all SentencePiece input files must exist')
    sentencepiece = require('sentencepiece', 'reparos-train')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / 'tokenizer'
    kwargs: dict[str, object] = {
        'input': ','.join(str(path) for path in files),
        'model_prefix': str(prefix),
        'vocab_size': settings.vocab_size,
        'model_type': settings.model_type,
        'character_coverage': settings.character_coverage,
        'normalization_rule_name': settings.normalization_rule,
        'pad_id': 0, 'bos_id': 1, 'eos_id': 2, 'unk_id': 3,
        'hard_vocab_limit': False,
    }
    if settings.input_sentence_size > 0:
        kwargs['input_sentence_size'] = settings.input_sentence_size
        kwargs['shuffle_input_sentence'] = True
    else:
        kwargs['shuffle_input_sentence'] = False
    sentencepiece.SentencePieceTrainer.train(**kwargs)
    manifest = {
        'schema_version': 1,
        'config': settings.__dict__,
        'training_inputs': fingerprint_files(files),
        'model_sha256': sha256_file(prefix.with_suffix('.model')),
        'vocab_sha256': sha256_file(prefix.with_suffix('.vocab')),
    }
    write_json(root / 'tokenizer-manifest.json', manifest)
    return manifest


class SentencePieceTokenizer:
    def __init__(self, model: str | Path) -> None:
        sentencepiece = require('sentencepiece', 'reparos-train')
        self.path = Path(model)
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(self.path))
        self.pad_id = int(self.processor.pad_id())
        self.bos_id = int(self.processor.bos_id())
        self.eos_id = int(self.processor.eos_id())
        self.unk_id = int(self.processor.unk_id())
        if (self.pad_id, self.bos_id, self.eos_id, self.unk_id) not in {(0, 1, 2, 3), (0, 2, 3, 1)}:
            raise ValueError(f'unexpected SentencePiece special token ids: pad={self.pad_id}, bos={self.bos_id}, eos={self.eos_id}, unk={self.unk_id}')

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def encode_batch(self, texts: list[str], add_bos: bool = True, add_eos: bool = True) -> list[list[int]]:
        all_ids = self.processor.encode(texts, out_type=int)
        bos = [self.bos_id] if add_bos else []
        eos = [self.eos_id] if add_eos else []
        return [bos + list(ids) + eos for ids in all_ids]

    def decode(self, ids: Iterable[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        return str(self.processor.decode([int(item) for item in ids if int(item) not in specials]))
