from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 8000
    model_type: str = 'unigram'
    character_coverage: float = 1.0
    normalization_rule: str = 'nmt_nfkc_cf'

    def validate(self) -> None:
        if self.vocab_size < 32:
            raise ValueError('vocab_size must be at least 32')
        if self.model_type not in {'unigram', 'bpe', 'char', 'word'}:
            raise ValueError('unsupported SentencePiece model type')


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8000
    hidden_size: int = 128
    attention_heads: int = 8
    encoder_layers: int = 1
    decoder_layers: int = 1
    feedforward_size: int = 512
    dropout: float = 0.1
    max_length: int = 64
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    def validate(self) -> None:
        if self.hidden_size % self.attention_heads:
            raise ValueError('hidden_size must be divisible by attention_heads')
        if min(self.encoder_layers, self.decoder_layers, self.max_length) < 1:
            raise ValueError('layer counts and max_length must be positive')
        if len({self.pad_id, self.bos_id, self.eos_id, self.unk_id}) != 4:
            raise ValueError('special token ids must be distinct')


@dataclass(frozen=True)
class TrainingConfig:
    stage: str = 'base'
    seed: int = 2026
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1.0
    adam_beta1: float = 0.8
    adam_beta2: float = 0.998
    adam_epsilon: float = 1e-8
    gradient_clip: float = 1.0
    warmup_steps: int = 4000
    label_smoothing: float = 0.0
    device: str = 'auto'

    def validate(self) -> None:
        if self.stage not in {'base', 'c1', 'c2'}:
            raise ValueError('stage must be base, c1, or c2')
        if min(self.epochs, self.batch_size) < 1:
            raise ValueError('epochs and batch_size must be positive')
        if self.learning_rate <= 0 or self.warmup_steps < 0:
            raise ValueError('learning rate must be positive and warmup non-negative')

    @classmethod
    def for_stage(cls, stage: str, **overrides: object) -> 'TrainingConfig':
        values: dict[str, object] = {'stage': stage, 'learning_rate': 1.0 if stage == 'base' else 0.0001}
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ReparoSConfig:
    profile: str = 'paper-reproduction'
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
