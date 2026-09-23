"""
Evaluate Arm G (ViT5-base fine-tuned) on local machine.
Tham khảo logic từ notebook/benchmark_arm_g_vit5_kaggle.ipynb

Requirements: ctranslate2, sentencepiece (đã có trong .venv)
ViT5 sử dụng T5Tokenizer (SentencePiece) — tokenizer.model nằm trong checkpoint-10000.
"""

import json
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CT2_DIR   = REPO_ROOT / "artifacts/arm_g_vit5_base/ct2_model"
SPM_MODEL = REPO_ROOT / "artifacts/arm_g_vit5_base/checkpoint-10000/spiece.model"
EVAL_DIR  = REPO_ROOT / "data/base_v3_eval"
OUT_DIR   = REPO_ROOT / "artifacts/arm_g_vit5_base"

assert CT2_DIR.exists() and (CT2_DIR / "model.bin").exists(), f"CT2 model not found: {CT2_DIR}"
assert SPM_MODEL.exists(), f"spiece.model not found: {SPM_MODEL}"

SUITES = {
    "plasticity":         (EVAL_DIR / "plasticity.src",         EVAL_DIR / "plasticity.tgt"),
    "retention":          (EVAL_DIR / "retention.src",          EVAL_DIR / "retention.tgt"),
    "user_centric":       (EVAL_DIR / "user_centric.src",       EVAL_DIR / "user_centric.tgt"),
    "protection_seen":    (EVAL_DIR / "protection_seen.src",    EVAL_DIR / "protection_seen.tgt"),
    "protection_heldout": (EVAL_DIR / "protection_heldout.src", EVAL_DIR / "protection_heldout.tgt"),
}

# 18 zero-click cases chuẩn hoá Canonical (sạch nhãn, nhất quán hợp đồng viết tắt)
ZERO_CLICK_CASES = [
    ("cau vuot song than",                          "cầu vượt sóng thần"),
    ("cho ba chieu",                                "chợ bà chiểu"),
    ("nga 4 hang xanh",                             "ngã 4 hàng xanh"),
    ("nga 3 vung tau",                              "ngã 3 vũng tàu"),
    ("bv cho ray",                                  "bệnh viện chợ rẫy"),
    ("Bv nhi đong",                                 "bệnh viện nhi đồng"),
    ("benh vien 175",                               "bệnh viện 175"),
    ("dh kinh te tphcm",                            "đại học kinh tế thành phố hồ chí minh"),
    ("86 xo viet nghe tinh p19 binh thanh",         "86 xô viết nghệ tĩnh phường 19 bình thạnh"),
    ("duong le van viet q9",                        "đường lê văn việt quận 9"),
    ("hem 212 thoai ngoc hau phuong phu thanh",     "hẻm 212 thoại ngọc hầu phường phú thạnh"),
    ("kcn song than 1",                             "khu công nghiệp sóng thần 1"),
    ("ubnd xa phuoc thai",                          "ủy ban nhân dân xã phước thái"),
    ("158/16 binh quew",                            "158/16 bình quới"),
    ("chung cu ha",                                 "chung cư hà"),
    ("ngã 6 tahnhf",                                "ngã 6 thành"),
    ("tttm aeon mall tan phu",                      "trung tâm thương mại aeon mall tân phú"),
    ("dh bach khoa ha noi",                         "đại học bách khoa hà nội"),
]

# Arm A-F baselines (từ notebook gốc)
BASELINES = {
    "Arm A (2E1D d128)":  {"enc_dec": "2E/1D",  "params": "6.5M",  "p50": "2.38ms", "mean_acc": 63.50, "plasticity": 41.7, "retention": 70.2, "user_centric": 58.3, "protection_heldout": 56.8, "zc_b1": 77.8},
    "Arm E (2E2D d128)":  {"enc_dec": "2E/2D",  "params": "7.1M",  "p50": "5.71ms", "mean_acc": 67.33, "plasticity": 43.6, "retention": 73.0, "user_centric": 66.7, "protection_heldout": 62.4, "zc_b1": 88.9},
    "Arm B (4E4D d128)":  {"enc_dec": "4E/4D",  "params": "9.6M",  "p50": "5.51ms", "mean_acc": 67.11, "plasticity": 43.7, "retention": 74.2, "user_centric": 66.7, "protection_heldout": 61.0, "zc_b1": 77.8},
    "Arm D (4E1D d128)":  {"enc_dec": "4E/1D",  "params": "7.7M",  "p50": "7.25ms", "mean_acc": 65.87, "plasticity": 42.4, "retention": 71.1, "user_centric": 66.7, "protection_heldout": 58.4, "zc_b1": 77.8},
    "Arm F (2E1D d256)":  {"enc_dec": "2E/1D",  "params": "13.4M", "p50": "3.97ms", "mean_acc": 64.76, "plasticity": 42.4, "retention": 72.5, "user_centric": 58.3, "protection_heldout": 58.8, "zc_b1": 77.8},
    "Arm C (6E6D d128)":  {"enc_dec": "6E/6D",  "params": "12.1M", "p50": "17.63ms","mean_acc": 64.23, "plasticity": 42.7, "retention": 72.8, "user_centric": 58.3, "protection_heldout": 59.6, "zc_b1": 88.9},
}

# ── Load tokenizer (SentencePiece, giống ViT5/T5Tokenizer) ──────────────────
print(f"Loading SentencePiece tokenizer: {SPM_MODEL}")
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load(str(SPM_MODEL))
print(f"  Vocab size: {sp.get_piece_size()}")

# T5/ViT5 huấn luyện với </s> ở cuối MỌI source. sp.encode_as_pieces() không tự
# thêm (khác với tokenizer(...) của HuggingFace), và nếu thiếu thì decoder không
# bao giờ nhận được tín hiệu dừng đã học -> sinh lặp vô hạn tới max length.
EOS = "</s>"
MAX_DECODING_LENGTH = 64

# spiece.model chỉ có 36.000 piece; </s> là control token do T5 thêm vào và chỉ
# tồn tại trong vocab 36.096 của CTranslate2. Kiểm tra ở đó, không phải ở spm.
with open(CT2_DIR / "shared_vocabulary.json", encoding="utf-8") as f:
    _ct2_vocab = json.load(f)
assert EOS in _ct2_vocab, f"{EOS} không có trong shared_vocabulary.json của CT2"


def encode_source(text):
    return sp.encode_as_pieces(text.strip().lower()) + [EOS]


def decode_hypothesis(pieces):
    # CTranslate2 thường đã cắt </s>; lọc lại cho chắc để </s> không lọt vào
    # chuỗi so khớp exact-match.
    return sp.decode_pieces([p for p in pieces if p != EOS]).strip().lower()


# ── Load CTranslate2 Translator (CPU INT8, 4 threads) ───────────────────────
print(f"\nLoading CTranslate2 model: {CT2_DIR}")
import ctranslate2
translator = ctranslate2.Translator(
    str(CT2_DIR), device="cpu", compute_type="int8", intra_threads=4
)
print("  Model loaded OK")

# Warmup
translator.translate_batch(
    [encode_source("ha noi")], beam_size=1, max_decoding_length=MAX_DECODING_LENGTH
)
print("  Warmup done\n")


# ── Helpers ──────────────────────────────────────────────────────────────────
def predict_batch(texts, beam_size=1, rep_penalty=1.2, batch_size=64):
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        token_batch = [encode_source(t) for t in chunk]
        outputs = translator.translate_batch(
            token_batch,
            beam_size=beam_size,
            repetition_penalty=rep_penalty,
            max_decoding_length=MAX_DECODING_LENGTH,
        )
        for out in outputs:
            preds.append(decode_hypothesis(out.hypotheses[0]))
    return preds


def compute_acc(preds, tgts):
    assert len(preds) == len(tgts)
    c = sum(1 for p, t in zip(preds, tgts) if p.strip().lower() == t.strip().lower())
    return (c / len(tgts)) * 100.0


# ── Evaluate 5 Frozen Suites ─────────────────────────────────────────────────
print("=" * 70)
print("EVALUATING ARM G ON 5 FROZEN SUITES (3,150 queries)")
print("=" * 70)

suite_results = {}
suite_accs = []

for s_name, (src_path, tgt_path) in SUITES.items():
    srcs = [l.strip() for l in open(src_path, encoding="utf-8") if l.strip()]
    tgts = [l.strip() for l in open(tgt_path, encoding="utf-8") if l.strip()]
    t0 = time.perf_counter()
    preds = predict_batch(srcs, beam_size=1, rep_penalty=1.2)
    dur = time.perf_counter() - t0
    acc = compute_acc(preds, tgts)
    lat_ms = (dur / len(srcs)) * 1000.0
    suite_results[s_name] = {"acc": acc, "lat_ms": lat_ms, "n": len(srcs)}
    suite_accs.append(acc)
    print(f"  {s_name:<22}: {acc:6.2f}%  ({len(srcs):>4} pairs | {lat_ms:.2f}ms/query)")

mean_acc = sum(suite_accs) / len(suite_accs)
print(f"\n  >>> 5-SUITE MEAN ACCURACY (ARM G): {mean_acc:.2f}%")

# ── Evaluate Zero-Click (18 cases) ───────────────────────────────────────────
print("\n" + "=" * 70)
print("ZERO-CLICK EVALUATION (18 CASES)")
print("=" * 70)

zc_srcs = [c[0] for c in ZERO_CLICK_CASES]
zc_tgts = [c[1] for c in ZERO_CLICK_CASES]

zc_preds_b1 = predict_batch(zc_srcs, beam_size=1,  rep_penalty=1.2)
zc_preds_b4 = predict_batch(zc_srcs, beam_size=4,  rep_penalty=1.2)
zc_preds_b10 = predict_batch(zc_srcs, beam_size=10, rep_penalty=1.2)

zc_acc_b1  = compute_acc(zc_preds_b1,  zc_tgts)
zc_acc_b4  = compute_acc(zc_preds_b4,  zc_tgts)
zc_acc_b10 = compute_acc(zc_preds_b10, zc_tgts)

print(f"\n  Zero-Click Acc  |  Beam 1: {zc_acc_b1:.1f}%  |  Beam 4: {zc_acc_b4:.1f}%  |  Beam 10: {zc_acc_b10:.1f}%")

print("\n  Detail (Beam 1):")
print(f"  {'Input':<45} {'Expected':<45} {'Pred':<45} {'OK?'}")
print("  " + "-" * 140)
for inp, exp, pred in zip(zc_srcs, zc_tgts, zc_preds_b1):
    ok = "✅" if pred == exp.lower() else "❌"
    print(f"  {inp:<45} {exp:<45} {pred:<45} {ok}")

# ── Latency Profiling ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LATENCY PROFILING (CPU INT8, 4 threads)")
print("=" * 70)

test_queries = [
    "dh bach khoa ha noi",
    "86 xo viet nghe tinh p19 binh thanh",
    "bv cho ray",
    "duong le van viet q9",
    "kcn song than 1",
]

latency = {}
for beam in (1, 4):
    lats = []
    for q in test_queries * 20:
        toks = encode_source(q)
        t0 = time.perf_counter()
        translator.translate_batch(
            [toks],
            beam_size=beam,
            repetition_penalty=1.2,
            max_decoding_length=MAX_DECODING_LENGTH,
        )
        lats.append((time.perf_counter() - t0) * 1000.0)
    lats.sort()
    latency[beam] = (lats[len(lats) // 2], lats[int(len(lats) * 0.9)])
    print(f"  Beam {beam:>2}: P50 = {latency[beam][0]:.2f}ms | P90 = {latency[beam][1]:.2f}ms")

p50_b1, p90_b1 = latency[1]
p50_b4, p90_b4 = latency[4]

# ── Comparison Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 120)
print("FULL COMPARISON: ARM G (ViT5-base Pretrained) vs ARMS A–F (Scratch)")
print("=" * 120)

header = f"{'Model':<25} {'Arch':<10} {'Params':>8} {'Pretrained':>11} {'P50 CPU':>10} {'Mean Acc':>10} {'Plasticity':>11} {'Retention':>10} {'UserCentric':>12} {'PH':>8} {'ZC(B1)':>8}"
print(header)
print("-" * 120)

for arm_name, d in BASELINES.items():
    row = (
        f"  {arm_name:<23} {d['enc_dec']:<10} {d['params']:>8} {'No':>11} {d['p50']:>10}"
        f" {d['mean_acc']:>9.2f}%"
        f" {d['plasticity']:>10.1f}%"
        f" {d['retention']:>9.1f}%"
        f" {d['user_centric']:>11.1f}%"
        f" {d['protection_heldout']:>7.1f}%"
        f" {d['zc_b1']:>7.1f}%"
    )
    print(row)

g = suite_results
g_row = (
    f"  {'Arm G (ViT5-base)':<23} {'12E/12D':<10} {'226M':>8} {'YES ✅':>11} {f'{p50_b1:.2f}ms':>10}"
    f" {mean_acc:>9.2f}%"
    f" {g['plasticity']['acc']:>10.1f}%"
    f" {g['retention']['acc']:>9.1f}%"
    f" {g['user_centric']['acc']:>11.1f}%"
    f" {g['protection_heldout']['acc']:>7.1f}%"
    f" {zc_acc_b1:>7.1f}%"
)
print("-" * 120)
print(g_row)
print("=" * 120)

delta_vs_best_scratch = mean_acc - max(d["mean_acc"] for d in BASELINES.values())
delta_vs_arm_a = mean_acc - BASELINES["Arm A (2E1D d128)"]["mean_acc"]
print(f"\n  Arm G vs Best Scratch (Arm E 67.33%): {delta_vs_best_scratch:+.2f}%")
print(f"  Arm G vs Baseline Arm A (63.50%):     {delta_vs_arm_a:+.2f}%")
print(f"  Arm G Latency vs Arm A (2.38ms):      {p50_b1:.2f}ms  ({p50_b1/2.38:.1f}x slower)")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "arm": "arm_g_vit5_base",
    "model": "VietAI/vit5-base (fine-tuned 10k steps, 300K pairs)",
    "params": "226M",
    "ct2_model": str(CT2_DIR),
    "suites": suite_results,
    "mean_acc": mean_acc,
    "zero_click": {
        "acc_beam1":  zc_acc_b1,
        "acc_beam4":  zc_acc_b4,
        "acc_beam10": zc_acc_b10,
        "preds_beam1": {s: p for s, p in zip(zc_srcs, zc_preds_b1)},
    },
    "latency": {
        "cpu_int8_p50_beam1_ms": p50_b1,
        "cpu_int8_p90_beam1_ms": p90_b1,
        "cpu_int8_p50_beam4_ms": p50_b4,
        "cpu_int8_p90_beam4_ms": p90_b4,
    },
}

out_json = OUT_DIR / "benchmark_results_arm_g.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[+] Results saved to: {out_json}")
