from pathlib import Path
import json
from reparos.continual_finetune import evaluate_continual_splits
from reparos.serving.ctranslate2 import CTranslate2Predictor

def main():
    ct2_model = Path('artifacts/reparos-base-v2-32-final-1/ctranslate2')
    eval_root = Path('data/reparos/interleaved-continual-v1/eval')

    print('Loading predictor...')
    predictor = CTranslate2Predictor(ct2_model, device='cpu')
    print('Evaluating continual splits on Base V2...')
    res = evaluate_continual_splits(predictor, eval_root, batch_size=64)

    for split in ('plasticity', 'retention', 'protection_seen', 'protection_heldout'):
        ov = res[split]['_overall']
        acc = ov['exact_accuracy'] * 100
        noop = ov['noop_rate'] * 100
        fcr = ov['fcr'] * 100
        n = ov['count']
        print(f'Split {split:18s}: Acc={acc:5.2f}%, NoOp={noop:5.2f}%, FCR={fcr:5.2f}% (n={n})')

    # Save Base V2 baseline numbers
    out_file = Path('data/reparos/interleaved-continual-v1/base_v2_baseline_metrics.json')
    out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved base_v2_baseline_metrics.json successfully!')

if __name__ == '__main__':
    main()
