from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebook/train_reparos_base_v2_kaggle_offline.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True),
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"][0] = md("""# ReparoS Base V2-32 — complete offline experiment

Train a complete Base V2 from scratch: a new train-only SentencePiece tokenizer, new OpenNMT vocabularies and new model weights. The original Base keeps its original tokenizer during evaluation. Everything runs from the attached offline Kaggle Dataset.
""")

    config = "".join(notebook["cells"][2]["source"])
    replacements = {
        "TRAIN_STEPS = 75_000": "TRAIN_STEPS = 10_000",
        "VALID_STEPS = 2_500": "VALID_STEPS = 1_000",
        "CHECKPOINT_STEPS = 5_000": "CHECKPOINT_STEPS = 1_000",
        "BATCH_SIZE_TOKENS = 16_384": "BATCH_SIZE_TOKENS = 131_072",
        "BUCKET_SIZE = 32_768": "BUCKET_SIZE = 262_144",
        "NUM_WORKERS = 2": "NUM_WORKERS = 8",
    }
    for old, new in replacements.items():
        config = config.replace(old, new)
    config += "TOKENIZER_CORPUS_SIZE = 500_000\nTOKENIZER_VOCAB_SIZE = 8_000\n"
    notebook["cells"][2] = code(config)

    setup = "".join(notebook["cells"][4]["source"])
    marker = "os.chdir(REPO_ROOT)\n"
    compatibility_patch = r'''# CTranslate2 4.6.x does not expose the 4.8.x unsafe_deserialization
# constructor argument. Checkpoint trust is already explicitly required by CLI.
ct2_source = REPO_ROOT / 'src/reparos/serving/ctranslate2.py'
ct2_text = ct2_source.read_text(encoding='utf-8')
ct2_old = """    converter = ctranslate2.converters.OpenNMTPyConverter(
        str(checkpoint_path), unsafe_deserialization=trust_checkpoint,
    )
"""
ct2_new = """    try:
        converter = ctranslate2.converters.OpenNMTPyConverter(
            str(checkpoint_path), unsafe_deserialization=trust_checkpoint,
        )
    except TypeError as error:
        if 'unsafe_deserialization' not in str(error):
            raise
        if not trust_checkpoint:
            raise RuntimeError(
                'CTranslate2 4.6.x requires --trust-checkpoint for OpenNMT conversion'
            ) from error
        converter = ctranslate2.converters.OpenNMTPyConverter(str(checkpoint_path))
"""
if ct2_old in ct2_text:
    ct2_source.write_text(ct2_text.replace(ct2_old, ct2_new), encoding='utf-8')

'''
    assert marker in setup
    notebook["cells"][4] = code(setup.replace(marker, compatibility_patch + marker))

    locate = "".join(notebook["cells"][6]["source"])
    locate = locate.replace("TOKENIZER_MODEL = unique_match('tokenizer.model', 'tokenizer')", "ORIGINAL_TOKENIZER_MODEL = unique_match('tokenizer.model', 'original tokenizer')")
    locate = locate.replace("print('Tokenizer:', TOKENIZER_MODEL)", "print('Original tokenizer:', ORIGINAL_TOKENIZER_MODEL)")
    notebook["cells"][6] = code(locate)

    notebook["cells"][7] = md("""## 4. Train a new tokenizer, build new vocabularies and configure Base V2

The tokenizer corpus contains every accepted clean training query plus a deterministic reservoir sample of noisy training queries, capped at 500K total lines. Validation and test are never used. OpenNMT vocabularies are then rebuilt from Base V2 with this tokenizer.
""")
    notebook["cells"][8] = code(r'''import random
from reparos.config import TokenizerConfig
from reparos.tokenization import train_sentencepiece

RUN_ROOT.mkdir(parents=True, exist_ok=True)

# Construct a balanced, deterministic train-only tokenizer corpus.
clean_lines = (BASE_V2_ROOT / 'train.clean.src').read_text(encoding='utf-8').splitlines()
noisy_budget = max(0, TOKENIZER_CORPUS_SIZE - len(clean_lines))
rng = random.Random(SEED)
noisy_sample = []
with (BASE_V2_ROOT / 'train.noisy.src').open(encoding='utf-8') as stream:
    for index, line in enumerate(stream):
        value = line.rstrip('\n\r')
        if index < noisy_budget:
            noisy_sample.append(value)
        else:
            replacement = rng.randint(0, index)
            if replacement < noisy_budget:
                noisy_sample[replacement] = value
tokenizer_lines = clean_lines + noisy_sample
rng.shuffle(tokenizer_lines)
TOKENIZER_CORPUS = RUN_ROOT / 'tokenizer-train.txt'
TOKENIZER_CORPUS.write_text('\n'.join(tokenizer_lines) + '\n', encoding='utf-8')
assert len(tokenizer_lines) == min(TOKENIZER_CORPUS_SIZE, len(clean_lines) + 5_410_493)

TOKENIZER_ROOT = RUN_ROOT / 'tokenizer-v2'
train_sentencepiece(
    [TOKENIZER_CORPUS], TOKENIZER_ROOT,
    TokenizerConfig(vocab_size=TOKENIZER_VOCAB_SIZE, input_sentence_size=0),
)
TOKENIZER_MODEL = TOKENIZER_ROOT / 'tokenizer.model'
assert TOKENIZER_MODEL.is_file()

def concatenate(destination, sources):
    with destination.open('wb') as target:
        for source in sources:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, target, length=16 * 1024 * 1024)

VALID_SRC = RUN_ROOT / 'validation.src'
VALID_TGT = RUN_ROOT / 'validation.tgt'
concatenate(VALID_SRC, [BASE_V2_ROOT / 'validation.noisy.src', BASE_V2_ROOT / 'validation.clean.src'])
concatenate(VALID_TGT, [BASE_V2_ROOT / 'validation.noisy.tgt', BASE_V2_ROOT / 'validation.clean.tgt'])

config = json.loads(ORIGINAL_CONFIG.read_text(encoding='utf-8'))
config.update({
    'save_data': str(RUN_ROOT / 'vocab'),
    'src_vocab': str(RUN_ROOT / 'vocab.src'),
    'tgt_vocab': str(RUN_ROOT / 'vocab.tgt'),
    'src_vocab_size': TOKENIZER_VOCAB_SIZE,
    'tgt_vocab_size': TOKENIZER_VOCAB_SIZE,
    'src_subword_model': str(TOKENIZER_MODEL),
    'tgt_subword_model': str(TOKENIZER_MODEL),
    'save_model': str(RUN_ROOT / 'reparos_base_v2'),
    'train_steps': TRAIN_STEPS,
    'valid_steps': VALID_STEPS,
    'save_checkpoint_steps': CHECKPOINT_STEPS,
    'batch_size': BATCH_SIZE_TOKENS,
    'bucket_size': BUCKET_SIZE,
    'num_workers': NUM_WORKERS,
    'keep_checkpoint': 10,
    'seed': SEED,
    'transformer_ff': 2048,
    'enc_layers': 2,
    'dec_layers': 1,
    'data': {
        'noisy': {
            'path_src': str(BASE_V2_ROOT / 'train.noisy.src'),
            'path_tgt': str(BASE_V2_ROOT / 'train.noisy.tgt'),
            'transforms': ['sentencepiece'], 'weight': 75,
        },
        'clean': {
            'path_src': str(BASE_V2_ROOT / 'train.clean.src'),
            'path_tgt': str(BASE_V2_ROOT / 'train.clean.tgt'),
            'transforms': ['sentencepiece'], 'weight': 25,
        },
        'valid': {
            'path_src': str(VALID_SRC), 'path_tgt': str(VALID_TGT),
            'transforms': ['sentencepiece'],
        },
    },
})
config.pop('train_from', None)

def checkpoint_step(path):
    return int(path.stem.rsplit('_step_', 1)[1])

existing = sorted(RUN_ROOT.glob('reparos_base_v2_step_*.pt'), key=checkpoint_step)
if existing and RESUME_IF_AVAILABLE:
    config['train_from'] = str(existing[-1])
    assert checkpoint_step(existing[-1]) < TRAIN_STEPS, (existing[-1], TRAIN_STEPS)
    print('Resuming from:', existing[-1])
elif existing:
    raise FileExistsError(f'Existing checkpoints found: {existing}')
else:
    print('Fresh Base V2 training with a new tokenizer and vocabulary')

CONFIG_PATH = RUN_ROOT / 'opennmt-base-v2.json'
CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

# A fresh run must build OpenNMT token-frequency vocabularies with the new tokenizer.
if not existing:
    subprocess.run([
        str(VENV_PY), '-m', 'onmt.bin.build_vocab',
        '-config', str(CONFIG_PATH), '-n_sample', '-1',
    ], check=True)
assert (RUN_ROOT / 'vocab.src').is_file() and (RUN_ROOT / 'vocab.tgt').is_file()
print('Tokenizer:', TOKENIZER_MODEL)
print('Tokenizer corpus rows:', len(tokenizer_lines))
print('Config:', CONFIG_PATH)
''')

    evaluation = "".join(notebook["cells"][12]["source"])
    old_loop = """for system, checkpoint in (
    ('base', ORIGINAL_BASE_CHECKPOINT),
    ('base_v2_32', BASE_V2_CHECKPOINT),
):
"""
    new_loop = """for system, checkpoint, tokenizer in (
    ('base', ORIGINAL_BASE_CHECKPOINT, ORIGINAL_TOKENIZER_MODEL),
    ('base_v2_32', BASE_V2_CHECKPOINT, TOKENIZER_MODEL),
):
"""
    if old_loop in evaluation:
        evaluation = evaluation.replace(old_loop, new_loop)
        evaluation = evaluation.replace("tokenizer=TOKENIZER_MODEL,", "tokenizer=tokenizer,")
        notebook["cells"][12] = code(evaluation)

    package = "".join(notebook["cells"][18]["source"])
    if "vocab.src" not in package:
        package = package.replace(
            "shutil.copy2(TOKENIZER_MODEL, EXPORT_ROOT / 'tokenizer.model')",
            "shutil.copy2(TOKENIZER_MODEL, EXPORT_ROOT / 'tokenizer.model')\nshutil.copy2(TOKENIZER_ROOT / 'tokenizer.vocab', EXPORT_ROOT / 'tokenizer.vocab')\nshutil.copy2(TOKENIZER_ROOT / 'tokenizer-manifest.json', EXPORT_ROOT / 'tokenizer-manifest.json')\nshutil.copy2(RUN_ROOT / 'vocab.src', EXPORT_ROOT / 'vocab.src')\nshutil.copy2(RUN_ROOT / 'vocab.tgt', EXPORT_ROOT / 'vocab.tgt')",
        )
        notebook["cells"][18] = code(package)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{NOTEBOOK}:cell-{index}")
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK, len(notebook["cells"]), "cells")


if __name__ == "__main__":
    main()
