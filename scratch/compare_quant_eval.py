import sys, time, unicodedata
from pathlib import Path
import ctranslate2, sentencepiece as spm

def norm(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text.strip().lower()).split())

eval_dir = Path('data/base_v3_eval')
suites = ['plasticity', 'retention', 'protection_seen', 'protection_heldout', 'user_centric']

all_samples = []
for s in suites:
    with open(eval_dir / f'{s}.src', encoding='utf-8') as fs, open(eval_dir / f'{s}.tgt', encoding='utf-8') as ft:
        for src, tgt in zip(fs, ft):
            all_samples.append((s, norm(src), norm(tgt)))

print(f'Total eval queries loaded: {len(all_samples):,}')

model_dir = Path('artifacts/reparos_base_v3_production_checkpoints/checkpoints/base_v3_production/ctranslate2_export')
sp = spm.SentencePieceProcessor(model_file=str(model_dir / 'tokenizer.model'))
tok_inputs = [sp.encode_as_pieces(s) for _, s, _ in all_samples]

results = {}
for ctype in ['int8', 'float32']:
    print(f'Evaluating compute_type={ctype}...')
    trans = ctranslate2.Translator(str(model_dir), device='cpu', compute_type=ctype, intra_threads=4)
    t0 = time.perf_counter()
    preds = trans.translate_batch(tok_inputs, beam_size=1, repetition_penalty=1.15, max_decoding_length=50)
    dur = time.perf_counter() - t0
    
    suite_correct = {s: 0 for s in suites}
    suite_total = {s: 0 for s in suites}
    total_correct = 0
    decoded_preds = []
    for (s, src, tgt), pred in zip(all_samples, preds):
        p_text = norm(sp.decode_pieces(pred.hypotheses[0]))
        decoded_preds.append(p_text)
        is_corr = (p_text == tgt)
        if is_corr:
            total_correct += 1
            suite_correct[s] += 1
        suite_total[s] += 1
        
    results[ctype] = {
        'total_acc': total_correct / len(all_samples) * 100,
        'suite_acc': {s: suite_correct[s] / suite_total[s] * 100 for s in suites},
        'duration_s': dur,
        'lat_per_query_ms': dur / len(all_samples) * 1000,
        'preds': decoded_preds
    }

print('\n' + '='*75)
print('SO SANH KET QUA EVALUATION: INT8 vs FLOAT32 (KHONG QUANTIZE)')
print('='*75)
print(f'{"Suite":<25} | {"INT8 Accuracy":<15} | {"FLOAT32 Accuracy":<18} | {"Chenh lech":<10}')
print('-'*75)
for s in suites:
    acc_int8 = results['int8']['suite_acc'][s]
    acc_f32 = results['float32']['suite_acc'][s]
    diff = acc_f32 - acc_int8
    print(f'{s:<25} | {acc_int8:>13.2f}% | {acc_f32:>16.2f}% | {diff:>+8.2f}%')

print('-'*75)
tot_int8 = results['int8']['total_acc']
tot_f32 = results['float32']['total_acc']
diff_tot = tot_f32 - tot_int8
print(f'{"TONG CONG (3,100)":<25} | {tot_int8:>13.2f}% | {tot_f32:>16.2f}% | {diff_tot:>+8.2f}%')
print('='*75)

diff_cases = sum(1 for p1, p2 in zip(results['int8']['preds'], results['float32']['preds']) if p1 != p2)
print(f'So cau co output lech nhau giua INT8 va FLOAT32: {diff_cases}/{len(all_samples)} ({diff_cases/len(all_samples)*100:.2f}%)')
print(f'Toc do INT8:    {results["int8"]["lat_per_query_ms"]:.2f} ms/query ({results["int8"]["duration_s"]:.1f}s)')
print(f'Toc do FLOAT32: {results["float32"]["lat_per_query_ms"]:.2f} ms/query ({results["float32"]["duration_s"]:.1f}s)')
