"""
Diagnose WHERE the accuracy gap lives, per capability — và đo trần lý thuyết của
Knowledge Distillation trước khi bỏ tiền train teacher.

Trả lời 3 câu hỏi mà bảng mean-accuracy hiện tại KHÔNG trả lời được:

  1. Mean acc gộp 5 suite che mất phân bố. Suite nào, capability nào thực sự
     tạo ra chênh lệch giữa model to và model nhỏ?
  2. Model sai KIỂU GÌ? (phá câu sạch / không sửa gì / sai đường-phố / sai dấu)
     Mỗi kiểu lỗi có cách chữa khác nhau và chi phí khác nhau.
  3. TRẦN CỦA DISTILLATION: số ca teacher đúng mà student sai. Student sau khi
     distill KHÔNG THỂ vượt quá con số này. Nếu trần thấp -> distill vô nghĩa.

Usage:
    .venv/Scripts/python.exe -X utf8 scripts/diagnose_capability_gap.py
    .venv/Scripts/python.exe -X utf8 scripts/diagnose_capability_gap.py --models arm_a arm_e arm_g --teacher arm_g --student arm_e
"""

import argparse
import json
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "data/base_v3_eval"
PILOT_DIR = REPO_ROOT / "artifacts/pilot_scaling"
TOK_V3 = REPO_ROOT / "data/tokenizer_v3/tokenizer.model"
ARM_G_DIR = REPO_ROOT / "artifacts/arm_g_vit5_base"
OUT_JSON = REPO_ROOT / "artifacts/capability_gap_diagnosis.json"

SUITES = ["plasticity", "retention", "user_centric", "protection_seen", "protection_heldout"]

# Arms A-F: OpenNMT + tokenizer_v3 (12k SPM), source KHÔNG có </s>.
# Arm G: ViT5 + spiece.model riêng (36k SPM), source BẮT BUỘC có </s>.
MODELS = {
    "arm_a": dict(label="Arm A (2E1D d128)", ct2=PILOT_DIR / "arm_a_2e1d_d128/ct2_model", spm=TOK_V3, eos=False),
    "arm_b": dict(label="Arm B (4E4D d128)", ct2=PILOT_DIR / "arm_b_4e4d_d128/ct2_model", spm=TOK_V3, eos=False),
    "arm_c": dict(label="Arm C (6E6D d128)", ct2=PILOT_DIR / "arm_c_6e6d_d128/ct2_model", spm=TOK_V3, eos=False),
    "arm_d": dict(label="Arm D (4E1D d128)", ct2=PILOT_DIR / "arm_d_4e1d_d128/ct2_model", spm=TOK_V3, eos=False),
    "arm_e": dict(label="Arm E (2E2D d128)", ct2=PILOT_DIR / "arm_e_2e2d_d128/ct2_model", spm=TOK_V3, eos=False),
    "arm_f": dict(label="Arm F (2E1D d256)", ct2=PILOT_DIR / "arm_f_2e1d_d256/ct2_model", spm=TOK_V3, eos=False),
    "arm_g": dict(label="Arm G (ViT5 226M)", ct2=ARM_G_DIR / "ct2_model", spm=ARM_G_DIR / "checkpoint-10000/spiece.model", eos=True),
}

EOS = "</s>"
MAX_DECODING_LENGTH = 64


# ── Vietnamese helpers ───────────────────────────────────────────────────────
def strip_diacritics(text):
    """bỏ dấu để phân biệt 'sai dấu' với 'sai chữ'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


STREET_TYPES = {"đường", "phố", "ngõ", "hẻm", "ngách", "đại lộ"}


def classify_error(src, pred, gold):
    """Phân loại kiểu lỗi — mỗi kiểu ứng với một hướng chữa khác nhau."""
    s, p, g = src.strip().lower(), pred.strip().lower(), gold.strip().lower()

    if g == s and p != s:
        # Input vốn đã đúng, model tự ý sửa -> phá câu sạch (protection failure)
        return "copy_broken"
    if p == s and g != s:
        # Model không làm gì cả trong khi phải chuẩn hóa
        return "no_op"

    pt, gt = p.split(), g.split()
    if len(pt) == len(gt):
        diff = [(a, b) for a, b in zip(pt, gt) if a != b]
        if diff and all(a in STREET_TYPES and b in STREET_TYPES for a, b in diff):
            # đường <-> phố: thông tin KHÔNG nằm trong input, phải tra gazetteer
            return "street_type"
        if diff and all(strip_diacritics(a) == strip_diacritics(b) for a, b in diff):
            return "diacritic_only"
        return "token_substitution"
    if len(pt) > len(gt):
        return "over_expansion"
    return "under_expansion"


# ── Inference ────────────────────────────────────────────────────────────────
def load_model(key):
    import ctranslate2
    import sentencepiece as spm

    cfg = MODELS[key]
    assert (cfg["ct2"] / "model.bin").exists(), f"Thiếu CT2 model: {cfg['ct2']}"
    assert cfg["spm"].exists(), f"Thiếu tokenizer: {cfg['spm']}"

    sp = spm.SentencePieceProcessor()
    sp.load(str(cfg["spm"]))
    tr = ctranslate2.Translator(str(cfg["ct2"]), device="cpu", compute_type="int8", intra_threads=4)
    tr.translate_batch([sp.encode_as_pieces("ha noi")], beam_size=1, max_decoding_length=MAX_DECODING_LENGTH)
    return tr, sp, cfg["eos"]


def predict(tr, sp, use_eos, texts, batch_size=64):
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        toks = [sp.encode_as_pieces(t.strip().lower()) + ([EOS] if use_eos else []) for t in chunk]
        for out in tr.translate_batch(
            toks, beam_size=1, repetition_penalty=1.2, max_decoding_length=MAX_DECODING_LENGTH
        ):
            preds.append(sp.decode_pieces([p for p in out.hypotheses[0] if p != EOS]).strip().lower())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["arm_a", "arm_e", "arm_g"], choices=list(MODELS))
    ap.add_argument("--teacher", default="arm_g", choices=list(MODELS))
    ap.add_argument("--student", default="arm_e", choices=list(MODELS))
    args = ap.parse_args()

    # ── Load eval items ──────────────────────────────────────────────────────
    items = []
    for suite in SUITES:
        for line in open(EVAL_DIR / f"{suite}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            r["suite"] = suite
            items.append(r)
    srcs = [r["input"] for r in items]
    golds = [r["expected"] for r in items]
    print(f"Loaded {len(items):,} eval items across {len(SUITES)} suites\n")

    # ── Run every requested model ────────────────────────────────────────────
    all_preds = {}
    for key in args.models:
        print(f"Running {MODELS[key]['label']} ...", end=" ", flush=True)
        tr, sp, use_eos = load_model(key)
        t0 = time.perf_counter()
        all_preds[key] = predict(tr, sp, use_eos, srcs)
        print(f"done in {time.perf_counter() - t0:.1f}s")
        del tr
    print()

    correct = {k: [p.strip().lower() == g.strip().lower() for p, g in zip(v, golds)] for k, v in all_preds.items()}

    # ── 1. Per-capability accuracy matrix ────────────────────────────────────
    caps = []
    for r in items:
        key = (r["suite"], r["capability"])
        if key not in caps:
            caps.append(key)

    print("=" * 110)
    print("1. PER-CAPABILITY ACCURACY — mean acc gộp 5 suite giấu phân bố này")
    print("=" * 110)
    hdr = f"{'suite':<20} {'capability':<34} {'n':>5}" + "".join(f"{MODELS[k]['label'][:16]:>18}" for k in args.models)
    print(hdr)
    print("-" * len(hdr))

    cap_table = {}
    for suite, cap in caps:
        idx = [i for i, r in enumerate(items) if r["suite"] == suite and r["capability"] == cap]
        row = {k: 100.0 * sum(correct[k][i] for i in idx) / len(idx) for k in args.models}
        cap_table[f"{suite}/{cap}"] = {"n": len(idx), **row}
        print(f"{suite:<20} {cap:<34} {len(idx):>5}" + "".join(f"{row[k]:>17.1f}%" for k in args.models))

    print("-" * len(hdr))
    overall = {k: 100.0 * sum(correct[k]) / len(items) for k in args.models}
    print(f"{'MICRO-AVERAGE (3,150)':<60}" + "".join(f"{overall[k]:>17.1f}%" for k in args.models))

    # ── 2. Error taxonomy ────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("2. ERROR TAXONOMY — model sai KIỂU GÌ (mỗi kiểu là một hướng chữa khác nhau)")
    print("=" * 110)
    tax = {}
    for k in args.models:
        c = Counter(
            classify_error(srcs[i], all_preds[k][i], golds[i]) for i in range(len(items)) if not correct[k][i]
        )
        tax[k] = dict(c)
        total = sum(c.values())
        print(f"\n  {MODELS[k]['label']}  —  {total} lỗi / {len(items)} ca")
        for kind, n in c.most_common():
            print(f"      {kind:<22} {n:>5}  ({100.0 * n / total:5.1f}% số lỗi)")

    # ── 3. Distillation ceiling ──────────────────────────────────────────────
    T, S = args.teacher, args.student
    print("\n" + "=" * 110)
    print(f"3. TRẦN CỦA DISTILLATION — teacher={MODELS[T]['label']}  student={MODELS[S]['label']}")
    print("=" * 110)
    if T not in correct or S not in correct:
        print("  (bỏ qua: teacher hoặc student không nằm trong --models)")
        return

    groups = defaultdict(list)
    for i in range(len(items)):
        groups[(correct[T][i], correct[S][i])].append(i)

    both_ok = len(groups[(True, True)])
    teacher_only = len(groups[(True, False)])   # <- ĐÂY là tất cả những gì KD có thể mua
    student_only = len(groups[(False, True)])   # <- KD có RỦI RO làm mất nhóm này
    both_bad = len(groups[(False, False)])
    n = len(items)

    print(f"  Teacher ✅ / Student ✅   {both_ok:>5}  ({100.0*both_ok/n:5.1f}%)   student đã có sẵn")
    print(f"  Teacher ✅ / Student ❌   {teacher_only:>5}  ({100.0*teacher_only/n:5.1f}%)   <<< TRẦN TUYỆT ĐỐI CỦA KD")
    print(f"  Teacher ❌ / Student ✅   {student_only:>5}  ({100.0*student_only/n:5.1f}%)   RỦI RO: KD có thể xóa nhóm này")
    print(f"  Teacher ❌ / Student ❌   {both_bad:>5}  ({100.0*both_bad/n:5.1f}%)   teacher KHÔNG biết -> KD bất lực")

    print(f"\n  Student hiện tại        : {overall[S]:.2f}%")
    print(f"  Trần nếu KD hoàn hảo    : {overall[S] + 100.0*teacher_only/n:.2f}%  (= chính điểm của teacher)")
    print(f"  Thực tế KD hấp thụ ~70% : {overall[S] + 0.7*100.0*teacher_only/n:.2f}%")
    print(f"  Nếu KD làm mất 1/3 nhóm student-only: {overall[S] + 0.7*100.0*teacher_only/n - 0.33*100.0*student_only/n:.2f}%")

    print("\n  Phân bố 'teacher đúng / student sai' theo capability — KD mua được gì:")
    by_cap = Counter(f"{items[i]['suite']}/{items[i]['capability']}" for i in groups[(True, False)])
    for cap, cnt in by_cap.most_common():
        print(f"      {cap:<52} {cnt:>4} ca  ({100.0*cnt/max(teacher_only,1):5.1f}% tổng lợi ích KD)")

    print("\n  Nhóm CẢ HAI ĐỀU SAI theo capability — vùng KHÔNG model nào chạm tới (cần data/luật):")
    by_cap_bad = Counter(f"{items[i]['suite']}/{items[i]['capability']}" for i in groups[(False, False)])
    for cap, cnt in by_cap_bad.most_common(8):
        print(f"      {cap:<52} {cnt:>4} ca")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_JSON.write_text(
        json.dumps(
            {
                "models": args.models,
                "overall_micro_acc": overall,
                "per_capability": cap_table,
                "error_taxonomy": tax,
                "distillation": {
                    "teacher": T,
                    "student": S,
                    "both_correct": both_ok,
                    "teacher_only": teacher_only,
                    "student_only": student_only,
                    "both_wrong": both_bad,
                    "kd_ceiling_pp": 100.0 * teacher_only / n,
                    "teacher_only_by_capability": dict(by_cap),
                    "both_wrong_by_capability": dict(by_cap_bad),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[+] Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
