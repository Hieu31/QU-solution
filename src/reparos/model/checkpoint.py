from __future__ import annotations

from pathlib import Path

from reparos.config import ModelConfig
from reparos.dependencies import require
from reparos.manifests import require_matching_checksum


def load_checkpoint(path: str | Path, tokenizer_model: str | Path, device: str = 'cpu'):
    torch = require('torch', 'reparos-train')
    from reparos.model.transformer import ReparoSTransformer
    state = torch.load(Path(path), map_location=device, weights_only=False)
    if state.get('schema_version') != 1:
        raise ValueError('unsupported ReparoS checkpoint schema')
    require_matching_checksum(tokenizer_model, state['tokenizer_sha256'])
    config = ModelConfig(**state['model_config'])
    model = ReparoSTransformer(config)
    model.load_state_dict(state['model_state'])
    model.to(device).eval()
    return model, state
