from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from reparos.config import ModelConfig, TrainingConfig
from reparos.dependencies import require
from reparos.manifests import sha256_file, write_json
from reparos.tokenization import SentencePieceTokenizer


def _parallel_lines(source: Path, target: Path) -> list[tuple[str, str]]:
    sources = source.read_text(encoding='utf-8').splitlines()
    targets = target.read_text(encoding='utf-8').splitlines()
    if not sources or len(sources) != len(targets):
        raise ValueError('parallel source and target must have equal non-zero lengths')
    return list(zip(sources, targets, strict=True))


def train_base(
    data: str | Path,
    tokenizer_model: str | Path,
    output: str | Path,
    *,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> dict[str, object]:
    torch = require('torch', 'reparos-train')
    from reparos.model.transformer import ReparoSTransformer

    settings = training_config or TrainingConfig.for_stage('base')
    settings.validate()
    if settings.stage != 'base':
        raise ValueError('train_base accepts only the base stage')
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    tokenizer = SentencePieceTokenizer(tokenizer_model)
    model_settings = model_config or ModelConfig(vocab_size=tokenizer.vocab_size)
    if model_settings.vocab_size != tokenizer.vocab_size:
        raise ValueError('model vocabulary size differs from tokenizer')
    model_settings.validate()
    data_root, output_root = Path(data), Path(output)
    train_rows = _parallel_lines(data_root / 'base' / 'train.src', data_root / 'base' / 'train.tgt')
    validation_rows = _parallel_lines(data_root / 'base' / 'validation.src', data_root / 'base' / 'validation.tgt')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if settings.device == 'auto' else settings.device
    device = torch.device(device_name)
    device_desc = f"{torch.cuda.get_device_name(device)} ({device})" if device.type == 'cuda' else f"{device}"
    print(f"[*] ReparoS Base Training starting on: {device_desc}")
    print(f"[*] Train samples: {len(train_rows):,}, Val samples: {len(validation_rows):,}")
    print(f"[*] Epochs: {settings.epochs}, Batch size: {settings.batch_size}")
    model = ReparoSTransformer(model_settings).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate,
        betas=(settings.adam_beta1, settings.adam_beta2), eps=settings.adam_epsilon,
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model_settings.pad_id, label_smoothing=settings.label_smoothing)
    output_root.mkdir(parents=True, exist_ok=True)
    history_path = output_root / 'training-history.jsonl'
    history_path.write_text('', encoding='utf-8')
    step = 0
    best_loss = math.inf
    total_batches = (len(train_rows) + settings.batch_size - 1) // settings.batch_size

    def batches(rows, epoch):
        order = list(range(len(rows)))
        random.Random(f'{settings.seed}:{epoch}').shuffle(order)
        for start in range(0, len(order), settings.batch_size):
            yield [rows[index] for index in order[start:start + settings.batch_size]]

    def tensors(rows):
        sources = [source for source, _ in rows]
        targets = [target for _, target in rows]
        encoded_source = [ids[:model_settings.max_length] for ids in tokenizer.encode_batch(sources)]
        encoded_target = [ids[:model_settings.max_length] for ids in tokenizer.encode_batch(targets)]
        def pad(values):
            width = max(map(len, values))
            return torch.tensor([value + [model_settings.pad_id] * (width - len(value)) for value in values], dtype=torch.long, device=device)
        return pad(encoded_source), pad(encoded_target)

    def validation_loss():
        model.eval()
        total, tokens = 0.0, 0
        with torch.no_grad():
            for batch in batches(validation_rows, 0):
                source, target = tensors(batch)
                logits = model(source, target[:, :-1])
                labels = target[:, 1:]
                count = int(labels.ne(model_settings.pad_id).sum())
                total += float(criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))) * count
                tokens += count
        return total / max(tokens, 1)

    def save(path: Path, epoch: int, val_loss: float):
        torch.save({
            'schema_version': 1, 'stage': 'base', 'epoch': epoch, 'step': step,
            'model_config': asdict(model_settings), 'training_config': asdict(settings),
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'validation_loss': val_loss,
            'tokenizer_sha256': sha256_file(tokenizer_model),
            'dataset': {
                'train_src_sha256': sha256_file(data_root / 'base' / 'train.src'),
                'train_tgt_sha256': sha256_file(data_root / 'base' / 'train.tgt'),
            },
        }, path)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_loss = total_tokens = 0
        batch_idx = 0
        for batch in batches(train_rows, epoch):
            batch_idx += 1
            source, target = tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target[:, :-1])
            labels = target[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            step += 1
            rate = settings.learning_rate
            if settings.warmup_steps:
                rate = settings.learning_rate * (model_settings.hidden_size ** -0.5) * min(step ** -0.5, step * settings.warmup_steps ** -1.5)
                for group in optimizer.param_groups:
                    group['lr'] = rate
            optimizer.step()
            count = int(labels.ne(model_settings.pad_id).sum())
            batch_loss = float(loss.detach())
            total_loss += batch_loss * count
            total_tokens += count
            if batch_idx % 200 == 0 or batch_idx == total_batches:
                cur_avg_loss = total_loss / max(total_tokens, 1)
                print(f"  [Epoch {epoch}/{settings.epochs}] Batch {batch_idx:,}/{total_batches:,} (Step {step:,}) | Loss: {batch_loss:.4f} (Avg: {cur_avg_loss:.4f}) | LR: {rate:.6f}")
        val_loss = validation_loss()
        epoch_train_loss = total_loss / max(total_tokens, 1)
        record = {'epoch': epoch, 'step': step, 'train_loss': epoch_train_loss, 'validation_loss': val_loss}
        with history_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        save(output_root / 'last.ckpt', epoch, val_loss)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            save(output_root / 'best.ckpt', epoch, val_loss)
        print(f"--> Epoch {epoch}/{settings.epochs} completed | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} {'[BEST]' if is_best else ''}")

    manifest = {
        'schema_version': 1, 'stage': 'base', 'device': device_name,
        'model_config': asdict(model_settings), 'training_config': asdict(settings),
        'best_validation_loss': best_loss,
        'best_checkpoint_sha256': sha256_file(output_root / 'best.ckpt'),
        'tokenizer': str(Path(tokenizer_model).resolve()),
    }
    write_json(output_root / 'manifest.json', manifest)
    return manifest
