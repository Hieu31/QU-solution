from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from reparos.manifests import fingerprint_files, sha256_file, write_json


@dataclass(frozen=True)
class DecodingConfig:
    beam_size: int = 10
    num_hypotheses: int = 10
    min_decoding_length: int = 0
    max_decoding_length: int = 100
    length_penalty: float = 1.0
    # OpenNMT-py `-length_penalty avg -alpha 1` and CTranslate2 use different
    # numeric conventions. CT2=0 reproduces OpenNMT average-score ranking for
    # converted OpenNMT-py checkpoints.
    ctranslate2_length_penalty: float = 0.0
    coverage_penalty: float = 0.0
    patience: float = 1.0
    no_repeat_ngram_size: int = 0
    disable_unk: bool = False

    def validate(self) -> None:
        if self.beam_size < 1 or not 1 <= self.num_hypotheses <= self.beam_size:
            raise ValueError('num_hypotheses must be between 1 and beam_size')
        if self.min_decoding_length < 0 or self.max_decoding_length < self.min_decoding_length:
            raise ValueError('invalid decoding length bounds')
        if self.length_penalty < 0 or self.ctranslate2_length_penalty < 0 or self.coverage_penalty < 0 or self.patience < 1:
            raise ValueError('invalid decoding penalty or patience')

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def resolved_base_architecture(
    opennmt_config: str | Path,
    tokenizer_model: str | Path,
    inputs: list[str | Path],
    decoding: DecodingConfig | None = None,
) -> dict[str, object]:
    config_path, tokenizer_path = Path(opennmt_config), Path(tokenizer_model)
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError('OpenNMT config and tokenizer model must exist')
    decode = decoding or DecodingConfig()
    decode.validate()
    state = json.loads(config_path.read_text(encoding='utf-8'))
    expected = {
        'enc_layers': 1, 'dec_layers': 1, 'heads': 8, 'hidden_size': 128,
        'optim': 'adam', 'learning_rate': 1.0, 'adam_beta1': 0.8,
        'adam_beta2': 0.998, 'adam_eps': 1e-8,
    }
    mismatches = {
        key: {'expected': value, 'actual': state.get(key)}
        for key, value in expected.items() if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f'config does not match published Base fields: {mismatches}')
    tokenizer_settings: dict[str, object] = {
        'vocab_size': 8000, 'model_type': 'unigram',
        'character_coverage': 1.0, 'normalization_rule': 'nmt_nfkc_cf',
    }
    tokenizer_manifest = tokenizer_path.parent / 'tokenizer-manifest.json'
    if tokenizer_manifest.is_file():
        tokenizer_settings.update(
            json.loads(tokenizer_manifest.read_text(encoding='utf-8')).get('config', {})
        )
    return {
        'schema_version': 1,
        'profile': 'reparos-2023-base-locked',
        'scope': 'ReparoS-Base without C1, C2, or the production ML ranker',
        'claim': 'paper-aligned published Base architecture with explicit local defaults',
        'published': {
            'task': 'sequence-to-sequence neural machine translation',
            'architecture': 'transformer-encoder-decoder',
            'encoder_layers': 1,
            'decoder_layers': 1,
            'attention_heads': 8,
            'hidden_size': 128,
            'tokenizer_family': 'SentencePiece',
            'vocabulary_size': 8000,
            'optimizer': 'adam',
            'learning_rate': 1.0,
            'adam_beta1': 0.8,
            'adam_beta2': 0.998,
            'adam_epsilon': 1e-8,
            'beam_size': 10,
            'training_toolkit': 'OpenNMT',
            'production_runtime': 'CTranslate2',
        },
        'inferred': {
            'word_embedding_size': 128,
            'position_encoding': 'sinusoidal-interleaved',
            'attention_type': 'scaled-dot',
            'decoder': 'autoregressive',
            'reason': {
                'word_embedding_size': 'matched to the published hidden dimension',
                'position_encoding': 'required by the disclosed Transformer architecture',
                'attention_type': 'original Transformer attention and official CTranslate2 compatibility',
                'decoder': 'implied by NMT formulation and beam decoding',
            },
        },
        'local_defaults': {
            'transformer_ff': state['transformer_ff'],
            'dropout': state['dropout'],
            'attention_dropout': state['attention_dropout'],
            'learning_rate_schedule': state['decay_method'],
            'warmup_steps': state['warmup_steps'],
            'gradient_clip': state['max_grad_norm'],
            'batch_type': state['batch_type'],
            'token_batch_size': state['batch_size'],
            'train_steps': state['train_steps'],
            'sentencepiece_model_type': tokenizer_settings['model_type'],
            'sentencepiece_character_coverage': tokenizer_settings['character_coverage'],
            'sentencepiece_normalization': tokenizer_settings['normalization_rule'],
            'shared_source_target_tokenizer': True,
            'shared_encoder_decoder_embeddings': False,
            'shared_decoder_output_embeddings': False,
            'decoding': decode.to_dict(),
        },
        'not_in_base_scope': {
            'c1_complex-error_finetuning': 'deferred',
            'c2_weak-feedback_finetuning': 'deferred',
            'production_ml_ranker': 'deferred and undisclosed by paper',
        },
        'artifacts': {
            'opennmt_config': str(config_path.resolve()),
            'opennmt_config_sha256': sha256_file(config_path),
            'tokenizer_model': str(tokenizer_path.resolve()),
            'tokenizer_sha256': sha256_file(tokenizer_path),
            'input_sha256': fingerprint_files(inputs),
        },
    }


def write_resolved_base_architecture(
    output: str | Path,
    opennmt_config: str | Path,
    tokenizer_model: str | Path,
    inputs: list[str | Path],
    decoding: DecodingConfig | None = None,
) -> Path:
    destination = Path(output)
    write_json(
        destination,
        resolved_base_architecture(opennmt_config, tokenizer_model, inputs, decoding),
    )
    return destination
