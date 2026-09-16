from __future__ import annotations

from reparos.dependencies import require

torch = require('torch', 'reparos-train')


def beam_search(model, source, beam_size: int = 10, max_length: int | None = None):
    if source.size(0) != 1:
        raise ValueError('beam_search currently accepts one query at a time')
    config = model.config
    limit = max_length or config.max_length
    memory, source_padding = model.encode(source)
    beams = [([config.bos_id], 0.0)]
    completed: list[tuple[list[int], float]] = []
    for _ in range(limit - 1):
        candidates: list[tuple[list[int], float]] = []
        for tokens, score in beams:
            if tokens[-1] == config.eos_id:
                completed.append((tokens, score))
                continue
            target = torch.tensor([tokens], dtype=torch.long, device=source.device)
            logits = model.decode(target, memory, source_padding)[:, -1]
            values, indices = torch.topk(torch.log_softmax(logits, dim=-1), beam_size)
            candidates.extend((tokens + [int(index)], score + float(value)) for value, index in zip(values[0], indices[0]))
        if not candidates:
            break
        beams = sorted(candidates, key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)[:beam_size]
        if len(completed) >= beam_size:
            break
    completed.extend(beams)
    return sorted(completed, key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)[:beam_size]
