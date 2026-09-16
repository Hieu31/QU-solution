from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from pathlib import Path

from reparos.architecture import DecodingConfig
from reparos.dependencies import require
from reparos.manifests import sha256_file, write_json


def export_opennmt_checkpoint(
    checkpoint: str | Path,
    tokenizer_model: str | Path,
    output: str | Path,
    *,
    quantization: str = 'float32',
    trust_checkpoint: bool = False,
) -> dict[str, object]:
    """Convert an OpenNMT-py checkpoint using CTranslate2's official converter.

    The internal reference checkpoint is intentionally rejected: CTranslate2
    conversion is architecture/toolkit specific and silent weight remapping is unsafe.
    """
    ctranslate2 = require('ctranslate2', 'reparos-serve')
    checkpoint_path, tokenizer_path, output_root = Path(checkpoint), Path(tokenizer_model), Path(output)
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError('checkpoint and tokenizer model must exist')
    import torch
    try:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        state = None
        if not trust_checkpoint:
            raise ValueError(
                'OpenNMT checkpoints contain pickled Python objects; pass '
                '--trust-checkpoint only for a checkpoint you created or trust'
            ) from error
    if isinstance(state, dict) and state.get('schema_version') == 1 and state.get('stage') == 'base':
        raise ValueError(
            'reference PyTorch checkpoints cannot be converted as OpenNMT; '
            'train/export an OpenNMT-py Base checkpoint for CTranslate2'
        )
    converter = ctranslate2.converters.OpenNMTPyConverter(
        str(checkpoint_path), unsafe_deserialization=trust_checkpoint,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    converter.convert(str(output_root), quantization=quantization, force=True)
    shutil.copy2(tokenizer_path, output_root / 'tokenizer.model')
    manifest = {
        'schema_version': 1,
        'source_format': 'OpenNMT-py',
        'source_checkpoint_sha256': sha256_file(checkpoint_path),
        'tokenizer_sha256': sha256_file(tokenizer_path),
        'ctranslate2_version': getattr(ctranslate2, '__version__', 'unknown'),
        'quantization': quantization,
        'unsafe_deserialization_authorized': trust_checkpoint,
    }
    write_json(output_root / 'conversion-manifest.json', manifest)
    return manifest


class CTranslate2Predictor:
    def __init__(self, model: str | Path, device: str = 'cpu', compute_type: str = 'default') -> None:
        ctranslate2 = require('ctranslate2', 'reparos-serve')
        from reparos.tokenization import SentencePieceTokenizer
        self.root = Path(model)
        self.tokenizer = SentencePieceTokenizer(self.root / 'tokenizer.model')
        self.translator = ctranslate2.Translator(str(self.root), device=device, compute_type=compute_type)

    def predict(
        self,
        text: str,
        beam_size: int = 10,
        num_hypotheses: int = 10,
        max_decoding_length: int | None = None,
        decoding: DecodingConfig | None = None,
    ) -> dict[str, object]:
        import time
        started = time.perf_counter()
        pieces = self.tokenizer.processor.encode(text, out_type=str)
        settings = decoding or DecodingConfig(
            beam_size=beam_size,
            num_hypotheses=num_hypotheses,
            max_decoding_length=max_decoding_length or 100,
        )
        settings.validate()
        result = self.translator.translate_batch(
            [pieces], beam_size=settings.beam_size,
            patience=settings.patience,
            num_hypotheses=settings.num_hypotheses,
            length_penalty=settings.ctranslate2_length_penalty,
            coverage_penalty=settings.coverage_penalty,
            no_repeat_ngram_size=settings.no_repeat_ngram_size,
            disable_unk=settings.disable_unk,
            min_decoding_length=settings.min_decoding_length,
            max_decoding_length=settings.max_decoding_length,
            return_scores=True,
        )[0]
        hypotheses = [self.tokenizer.processor.decode(tokens) for tokens in result.hypotheses]
        return {
            'input_query': text, 'top1_query': hypotheses[0],
            'hypotheses': hypotheses, 'sequence_scores': list(result.scores),
            'changed': hypotheses[0] != text,
            'latency_ms': (time.perf_counter() - started) * 1000,
            'model_stage': 'base', 'backend': 'ctranslate2',
        }
