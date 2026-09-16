from __future__ import annotations

import json
import tempfile
from pathlib import Path

from reparos.architecture import DecodingConfig
from reparos.manifests import write_json
from reparos.serving.ctranslate2 import CTranslate2Predictor
from reparos.training.opennmt import translate_opennmt


def compare_opennmt_ctranslate2(
    checkpoint: str | Path,
    ctranslate2_model: str | Path,
    tokenizer_model: str | Path,
    queries: str | Path,
    *,
    beam_size: int = 10,
    n_best: int = 10,
    output: str | Path | None = None,
    decoding: DecodingConfig | None = None,
) -> dict[str, object]:
    query_lines = [line.strip() for line in Path(queries).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not query_lines:
        raise ValueError('parity query file is empty')
    settings = decoding or DecodingConfig(beam_size=beam_size, num_hypotheses=n_best)
    settings.validate()
    with tempfile.TemporaryDirectory() as directory:
        reference_path = Path(directory) / 'opennmt.txt'
        translate_opennmt(
            checkpoint, tokenizer_model, queries, reference_path,
            decoding=settings,
        )
        lines = reference_path.read_text(encoding='utf-8').splitlines()
    expected = len(query_lines) * settings.num_hypotheses
    if len(lines) != expected:
        raise ValueError(f'OpenNMT returned {len(lines)} hypotheses; expected {expected}')
    reference = [
        lines[index:index + settings.num_hypotheses]
        for index in range(0, len(lines), settings.num_hypotheses)
    ]
    predictor = CTranslate2Predictor(ctranslate2_model)
    converted = [predictor.predict(query, decoding=settings)['hypotheses'] for query in query_lines]

    def normalized(value: str) -> str:
        return ' '.join(value.split())

    top1_matches = 0
    topk_set_matches = 0
    records = []
    for query, left, right in zip(query_lines, reference, converted, strict=True):
        left_norm = [normalized(item) for item in left]
        right_norm = [normalized(str(item)) for item in right]
        top1_equal = left_norm[0] == right_norm[0]
        topk_equal = set(left_norm) == set(right_norm)
        top1_matches += int(top1_equal)
        topk_set_matches += int(topk_equal)
        records.append({
            'query': query, 'top1_equal': top1_equal,
            'topk_set_equal': topk_equal,
            'opennmt': left_norm, 'ctranslate2': right_norm,
        })
    total = len(query_lines)
    report = {
        'queries': total,
        'decoding': settings.to_dict(),
        'top1_matches': top1_matches,
        'top1_parity': top1_matches / total,
        'topk_set_matches': topk_set_matches,
        'topk_set_parity': topk_set_matches / total,
        'records': records,
    }
    if output:
        write_json(output, report)
    return report
