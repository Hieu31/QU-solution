from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from reparos.architecture import DecodingConfig, write_resolved_base_architecture
from reparos.manifests import fingerprint_files, write_json


def _require_opennmt() -> None:
    try:
        import onmt  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            'OpenNMT-py is required; install with: uv sync --extra reparos-opennmt'
        ) from error


def build_opennmt_config(
    data: str | Path,
    tokenizer_model: str | Path,
    output: str | Path,
    *,
    train_steps: int = 100_000,
    valid_steps: int = 5_000,
    save_checkpoint_steps: int = 5_000,
    batch_size: int = 4096,
    bucket_size: int = 8192,
    num_workers: int = 2,
    transformer_ff: int = 512,
    dropout: float = 0.1,
    warmup_steps: int = 4000,
    gpu_rank: int | None = None,
) -> Path:
    if bucket_size < 1 or num_workers < 0 or transformer_ff < 1 or warmup_steps < 0:
        raise ValueError('invalid bucket, worker, FFN, or warmup setting')
    if not 0 <= dropout < 1:
        raise ValueError('dropout must be in [0, 1)')
    data_root, tokenizer, output_root = Path(data), Path(tokenizer_model), Path(output)
    files = {
        'train_src': data_root / 'base' / 'train.src',
        'train_tgt': data_root / 'base' / 'train.tgt',
        'valid_src': data_root / 'base' / 'validation.src',
        'valid_tgt': data_root / 'base' / 'validation.tgt',
        'tokenizer': tokenizer,
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing OpenNMT inputs: {missing}')
    output_root.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        'save_data': str(output_root / 'vocab'),
        'src_vocab': str(output_root / 'vocab.src'),
        'tgt_vocab': str(output_root / 'vocab.tgt'),
        'src_vocab_size': 8000,
        'tgt_vocab_size': 8000,
        'overwrite': True,
        'data': {
            'corpus_1': {
                'path_src': str(files['train_src']),
                'path_tgt': str(files['train_tgt']),
                'transforms': ['sentencepiece'],
                'weight': 1,
            },
            'valid': {
                'path_src': str(files['valid_src']),
                'path_tgt': str(files['valid_tgt']),
                'transforms': ['sentencepiece'],
            },
        },
        'src_subword_model': str(tokenizer),
        'tgt_subword_model': str(tokenizer),
        'save_model': str(output_root / 'reparos_base'),
        'encoder_type': 'transformer',
        'decoder_type': 'transformer',
        'enc_layers': 1,
        'dec_layers': 1,
        'heads': 8,
        'hidden_size': 128,
        'word_vec_size': 128,
        'transformer_ff': transformer_ff,
        'position_encoding': True,
        # OpenNMT-py 3.5 defaults to scaled-dot-flash, which its official
        # CTranslate2 converter does not support.
        'self_attn_type': 'scaled-dot',
        'dropout': [dropout],
        'attention_dropout': [dropout],
        'param_init': 0.0,
        'param_init_glorot': True,
        'optim': 'adam',
        'learning_rate': 1.0,
        'adam_beta1': 0.8,
        'adam_beta2': 0.998,
        'adam_eps': 1e-8,
        'decay_method': 'noam',
        'warmup_steps': warmup_steps,
        'max_grad_norm': 1.0,
        'batch_type': 'tokens',
        'batch_size': batch_size,
        'bucket_size': bucket_size,
        'num_workers': num_workers,
        'normalization': 'tokens',
        'train_steps': train_steps,
        'valid_steps': valid_steps,
        'save_checkpoint_steps': save_checkpoint_steps,
        'keep_checkpoint': 5,
        'seed': 2026,
        'world_size': 1,
        'gpu_ranks': [gpu_rank] if gpu_rank is not None else [],
    }
    path = output_root / 'opennmt-base.json'
    write_json(path, config)
    write_json(output_root / 'opennmt-input-manifest.json', {
        'schema_version': 1,
        'profile': 'reparos-base/openNMT-py',
        'input_sha256': fingerprint_files(files.values()),
        'published': {
            'enc_layers': 1, 'dec_layers': 1, 'heads': 8,
            'hidden_size': 128, 'learning_rate': 1.0,
            'adam_beta1': 0.8, 'adam_beta2': 0.998, 'adam_eps': 1e-8,
        },
        'local_defaults': {
            'transformer_ff': transformer_ff, 'dropout': dropout,
            'warmup_steps': warmup_steps,
            'batch_size_tokens': batch_size, 'bucket_size': bucket_size,
            'num_workers': num_workers, 'train_steps': train_steps,
        },
    })
    decoding = DecodingConfig()
    write_json(output_root / 'decoding-config.json', decoding.to_dict())
    write_resolved_base_architecture(
        output_root / 'resolved-architecture.json', path, tokenizer,
        [files['train_src'], files['train_tgt'], files['valid_src'], files['valid_tgt']],
        decoding,
    )
    return path


def build_opennmt_vocabulary(config: str | Path) -> None:
    _require_opennmt()
    config_path = Path(config)
    subprocess.run(
        [sys.executable, '-m', 'onmt.bin.build_vocab', '-config', str(config_path)],
        check=True,
    )
    # OpenNMT-py 3.5 writes CRLF on Windows but its vocabulary reader splits
    # raw bytes only on LF. The remaining CR makes the count fail `isdigit()`
    # and the full "token<TAB>count<CR>" becomes a token. Normalize to LF.
    state = json.loads(config_path.read_text(encoding='utf-8'))
    for key in ('src_vocab', 'tgt_vocab'):
        vocab = Path(str(state[key]))
        lines = vocab.read_text(encoding='utf-8').splitlines()
        with vocab.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines) + '\n')


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit('_step_', 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f'invalid OpenNMT checkpoint name: {path.name}') from error


def train_opennmt(config: str | Path) -> dict[str, object]:
    _require_opennmt()
    config_path = Path(config)
    build_opennmt_vocabulary(config_path)
    subprocess.run(
        [sys.executable, '-m', 'onmt.bin.train', '-config', str(config_path)],
        check=True,
    )
    state = json.loads(config_path.read_text(encoding='utf-8'))
    prefix = Path(str(state['save_model']))
    checkpoints = sorted(prefix.parent.glob(prefix.name + '_step_*.pt'), key=_checkpoint_step)
    if not checkpoints:
        raise RuntimeError('OpenNMT training completed without a checkpoint')
    result = {
        'config': str(config_path.resolve()),
        'checkpoints': [str(path.resolve()) for path in checkpoints],
        'latest_checkpoint': str(checkpoints[-1].resolve()),
    }
    write_json(config_path.parent / 'opennmt-training-result.json', result)
    return result


def translate_opennmt(
    checkpoint: str | Path,
    tokenizer_model: str | Path,
    queries: str | Path,
    output: str | Path,
    *,
    beam_size: int | None = None,
    n_best: int | None = None,
    decoding: DecodingConfig | None = None,
) -> Path:
    _require_opennmt()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    settings = decoding or DecodingConfig()
    settings.validate()
    resolved_beam = settings.beam_size if beam_size is None else beam_size
    resolved_n_best = settings.num_hypotheses if n_best is None else n_best
    if resolved_beam < 1 or not 1 <= resolved_n_best <= resolved_beam:
        raise ValueError('n_best must be between 1 and beam_size')
    command = [
        sys.executable, '-m', 'onmt.bin.translate',
        '-model', str(checkpoint), '-src', str(queries), '-output', str(destination),
        '-beam_size', str(resolved_beam), '-n_best', str(resolved_n_best),
        '-min_length', str(settings.min_decoding_length),
        '-max_length', str(settings.max_decoding_length),
        '-max_length_ratio', '0',
        '-length_penalty', 'avg', '-alpha', str(settings.length_penalty),
        '-coverage_penalty', 'none', '-beta', str(settings.coverage_penalty),
        '-block_ngram_repeat', str(settings.no_repeat_ngram_size),
        '-transforms', 'sentencepiece',
        '-src_subword_model', str(tokenizer_model),
        '-tgt_subword_model', str(tokenizer_model),
    ]
    if settings.disable_unk:
        command.append('-ban_unk_token')
    subprocess.run(command, check=True)
    return destination
