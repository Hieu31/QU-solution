from __future__ import annotations

import math

from reparos.config import ModelConfig
from reparos.dependencies import require

torch = require('torch', 'reparos-train')
nn = torch.nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_length: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        divisions = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        values = torch.zeros(max_length, hidden_size)
        values[:, 0::2] = torch.sin(positions * divisions)
        values[:, 1::2] = torch.cos(positions * divisions)
        self.register_buffer('values', values.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        return self.dropout(value + self.values[:, : value.size(1)])


class ReparoSTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_id)
        self.position = PositionalEncoding(config.hidden_size, config.max_length, config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            num_encoder_layers=config.encoder_layers,
            num_decoder_layers=config.decoder_layers,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.scale = math.sqrt(config.hidden_size)

    def encode(self, source):
        source_padding = source.eq(self.config.pad_id)
        embedded = self.position(self.embedding(source) * self.scale)
        return self.transformer.encoder(embedded, src_key_padding_mask=source_padding), source_padding

    def decode(self, target, memory, source_padding):
        target_padding = target.eq(self.config.pad_id)
        length = target.size(1)
        causal = torch.triu(torch.ones(length, length, device=target.device, dtype=torch.bool), diagonal=1)
        embedded = self.position(self.embedding(target) * self.scale)
        decoded = self.transformer.decoder(
            embedded, memory, tgt_mask=causal,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        return self.output(decoded)

    def forward(self, source, target_input):
        memory, source_padding = self.encode(source)
        return self.decode(target_input, memory, source_padding)
