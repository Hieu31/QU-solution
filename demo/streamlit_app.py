from __future__ import annotations

import difflib
import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import ctranslate2
import sentencepiece as spm
import streamlit as st
from st_keyup import st_keyup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODEL_CONFIGS = {
    "finetune_v2": {
        "title": "ReparoS Fine-Tune V2",
        "subtitle": "Curriculum-V2-Final · Vocab 8k",
        "ct2_dir": ROOT / "artifacts/reparos-curriculum-v2-final/ctranslate2",
        "tokenizer": ROOT / "artifacts/reparos-curriculum-v2-final/ctranslate2/tokenizer.model",
        "color": "#d97706",
        "bg_color": "#fffbeb",
        "border_color": "#f59e0b",
        "default_rep_penalty": 1.0,
    },
    "base_v3": {
        "title": "ReparoS Base V3",
        "subtitle": "Production Clean · Vocab 12k · 6.5M params",
        "ct2_dir": ROOT / "artifacts/checkpoints/base_v3_production/ctranslate2_export",
        "tokenizer": ROOT / "artifacts/checkpoints/base_v3_production/ctranslate2_export/tokenizer.model",
        "color": "#2563eb",
        "bg_color": "#eff6ff",
        "border_color": "#3b82f6",
        "default_rep_penalty": 1.2,
    },
    "continual_v1": {
        "title": "ReparoS Continual V1",
        "subtitle": "Continual Fine-Tune Final · Vocab 8k",
        "ct2_dir": ROOT / "artifacts/reparos-continual-v1-final/ctranslate2",
        "tokenizer": ROOT / "artifacts/reparos-continual-v1-final/ctranslate2/tokenizer.model",
        "color": "#059669",
        "bg_color": "#ecfdf5",
        "border_color": "#10b981",
        "default_rep_penalty": 1.0,
    },
}

MODEL_LOCK = threading.RLock()
FEEDBACK_LOCK = threading.RLock()
FEEDBACK_PATH = ROOT / "logs" / "demo_feedback.jsonl"


@st.cache_resource(show_spinner="Đang tải tokenizer & CTranslate2 model...")
def get_engine(ct2_dir: str, tokenizer_path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    # Automatically select compute_type float32 on CPU for compatibility
    translator = ctranslate2.Translator(str(ct2_dir), device="cpu", compute_type="float32")
    return sp, translator


def infer_single(
    sp: spm.SentencePieceProcessor,
    translator: ctranslate2.Translator,
    query: str,
    beam_size: int = 4,
    rep_penalty: float = 1.0,
) -> dict:
    if not query.strip():
        return {"top1": "", "hypotheses": [], "scores": [], "latency_ms": 0.0}

    t0 = time.perf_counter_ns()
    tokens = sp.encode_as_pieces(query.strip().lower())

    with MODEL_LOCK:
        kwargs = {
            "beam_size": beam_size,
            "num_hypotheses": min(beam_size, 5),
            "repetition_penalty": rep_penalty,
        }
        results = translator.translate_batch([tokens], **kwargs)

    lat_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    res = results[0]
    hyps = []
    scores = []
    for h in res.hypotheses:
        hyps.append(sp.decode_pieces(h))
    if hasattr(res, "scores") and res.scores:
        scores = [float(s) for s in res.scores]

    top1 = hyps[0] if hyps else ""
    return {
        "top1": top1,
        "hypotheses": hyps,
        "scores": scores,
        "latency_ms": lat_ms,
    }


def highlight_diff(s1: str, s2: str) -> tuple[str, str]:
    words1 = s1.split()
    words2 = s2.split()
    matcher = difflib.SequenceMatcher(None, words1, words2)

    h1_parts = []
    h2_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        sub1 = " ".join(words1[i1:i2])
        sub2 = " ".join(words2[j1:j2])
        if tag == "equal":
            h1_parts.append(sub1)
            h2_parts.append(sub2)
        elif tag == "replace":
            h1_parts.append(f'<span style="background-color: #fed7aa; color: #9a3412; font-weight: bold; padding: 2px 4px; border-radius: 4px;">{sub1}</span>')
            h2_parts.append(f'<span style="background-color: #bfdbfe; color: #1e40af; font-weight: bold; padding: 2px 4px; border-radius: 4px;">{sub2}</span>')
        elif tag == "delete":
            h1_parts.append(f'<span style="background-color: #fecaca; color: #991b1b; text-decoration: line-through; padding: 2px 4px; border-radius: 4px;">{sub1}</span>')
        elif tag == "insert":
            h2_parts.append(f'<span style="background-color: #bbf7d0; color: #166534; font-weight: bold; padding: 2px 4px; border-radius: 4px;">{sub2}</span>')

    return " ".join(h1_parts), " ".join(h2_parts)


def save_feedback(query: str, res_v2: dict, res_v3: dict, note: str = "") -> tuple[bool, str]:
    record = {
        "schema_version": 2,
        "feedback_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": query,
        "finetune_v2": {"top1": res_v2["top1"], "latency_ms": res_v2["latency_ms"]},
        "base_v3": {"top1": res_v3["top1"], "latency_ms": res_v3["latency_ms"]},
        "user_note": note,
    }
    with FEEDBACK_LOCK:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True, f"Đã lưu phản hồi · {record['feedback_id'][:8]}"


def main():
    st.set_page_config(
        page_title="ReparoS Model Comparator: Finetune V2 vs Base V3",
        page_icon="⚡",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { max-width: 1400px; padding-top: 1.5rem; }
        .answer-card {
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .answer-title {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .answer-text {
            font-size: 1.25rem;
            font-weight: 600;
            line-height: 1.4;
        }
        .candidate-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 8px;
            border-bottom: 1px solid #f1f5f9;
            font-size: 0.9rem;
        }
        .candidate-rank {
            font-weight: 700;
            color: #94a3b8;
            margin-right: 8px;
            min-width: 16px;
        }
        .candidate-text {
            flex-grow: 1;
            font-weight: 500;
        }
        .candidate-score {
            color: #64748b;
            font-size: 0.8rem;
            font-family: monospace;
        }
        .diff-box {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 8px;
            font-size: 1.05rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚡ ReparoS Lab: Fine-Tune V2 vs. Base V3")
    st.caption("So sánh thực nghiệm trực quan giữa **Finetune V2 (Curriculum-V2-Final)** và **Base V3 (Production CT2)**")

    # Check model artifacts
    for key in ["finetune_v2", "base_v3"]:
        cfg = MODEL_CONFIGS[key]
        if not cfg["ct2_dir"].exists():
            st.error(f"Không tìm thấy thư mục CTranslate2 của {cfg['title']} tại: `{cfg['ct2_dir']}`")
            st.stop()
        if not cfg["tokenizer"].exists():
            st.error(f"Không tìm thấy file tokenizer của {cfg['title']} tại: `{cfg['tokenizer']}`")
            st.stop()

    # Sidebar / Controls
    with st.sidebar:
        st.header("⚙️ Cấu hình Inference")
        beam_size = st.slider("Beam Size", min_value=1, max_value=8, value=4, step=1)
        v3_rep_penalty = st.slider(
            "Repetition Penalty (Base V3)",
            min_value=1.0,
            max_value=1.5,
            value=1.2,
            step=0.05,
            help="Hệ số phạt lặp từ cho Base V3 (khuyên dùng 1.2)",
        )
        show_continual = st.checkbox("Hiển thị thêm Continual V1 (3 cột)", value=False)
        debounce_ms = st.slider("Debounce gõ phím (ms)", min_value=100, max_value=800, value=300, step=50)

        st.divider()
        st.markdown("### 📌 Thông số 2 mô hình:")
        st.markdown(
            """
            - **Finetune V2**:
              - Vocab: 8,000 subwords
              - Arch: 2 Enc - 1 Dec, d128
              - Train: Curriculum v2 (60k steps)
            - **Base V3**:
              - Vocab: 12,000 subwords
              - Arch: 2 Enc - 1 Dec, d128 ff2048 (6.5M params)
              - Train: Base V3 recipe (4M clean pairs)
            """
        )

    # Preset queries for quick testing
    st.markdown("##### 🚀 Test nhanh các nhóm ca lỗi điển hình:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("📍 86 xo viet nghe tinh p19 binh thanh", use_container_width=True):
            st.session_state["query_input"] = "86 xo viet nghe tinh p19 binh thanh"
    with col_p2:
        if st.button("🏥 bv cho ray", use_container_width=True):
            st.session_state["query_input"] = "bv cho ray"
    with col_p3:
        if st.button("🔍 158/16 binh quew", use_container_width=True):
            st.session_state["query_input"] = "158/16 binh quew"
    with col_p4:
        if st.button("🏫 dh bach khoa ha noi", use_container_width=True):
            st.session_state["query_input"] = "dh bach khoa ha noi"

    col_p5, col_p6, col_p7, col_p8 = st.columns(4)
    with col_p5:
        if st.button("🏛️ ubnd q1", use_container_width=True):
            st.session_state["query_input"] = "ubnd q1"
    with col_p6:
        if st.button("🛣️ duong le thanh ton quan 1", use_container_width=True):
            st.session_state["query_input"] = "duong le thanh ton quan 1"
    with col_p7:
        if st.button("🏢 chung cu ha", use_container_width=True):
            st.session_state["query_input"] = "chung cu ha"
    with col_p8:
        if st.button("🏬 tttm vincom tran duy hung", use_container_width=True):
            st.session_state["query_input"] = "tttm vincom tran duy hung"

    default_query = st.session_state.get("query_input", "86 xo viet nghe tinh p19 binh thanh")

    query = st_keyup(
        "Nhập truy vấn địa điểm cần sửa lỗi:",
        value=default_query,
        debounce=debounce_ms,
        key="live_search_input",
    )

    if not query or not query.strip():
        st.info("Hãy nhập truy vấn hoặc click một nút test nhanh ở trên.")
        return

    # Load engines
    sp_v2, trans_v2 = get_engine(str(MODEL_CONFIGS["finetune_v2"]["ct2_dir"]), str(MODEL_CONFIGS["finetune_v2"]["tokenizer"]))
    sp_v3, trans_v3 = get_engine(str(MODEL_CONFIGS["base_v3"]["ct2_dir"]), str(MODEL_CONFIGS["base_v3"]["tokenizer"]))

    with st.spinner("Đang suy luận đồng thời..."):
        res_v2 = infer_single(sp_v2, trans_v2, query, beam_size=beam_size, rep_penalty=1.0)
        res_v3 = infer_single(sp_v3, trans_v3, query, beam_size=beam_size, rep_penalty=v3_rep_penalty)

        res_cont = None
        if show_continual and MODEL_CONFIGS["continual_v1"]["ct2_dir"].exists():
            sp_cont, trans_cont = get_engine(str(MODEL_CONFIGS["continual_v1"]["ct2_dir"]), str(MODEL_CONFIGS["continual_v1"]["tokenizer"]))
            res_cont = infer_single(sp_cont, trans_cont, query, beam_size=beam_size, rep_penalty=1.0)

    # Agreement / Disagreement Banner
    is_agree = (res_v2["top1"].strip().lower() == res_v3["top1"].strip().lower())
    if is_agree:
        st.success(f"✅ **Đồng thuận 100%**: Cả Finetune V2 và Base V3 đều dự đoán cùng một kết quả Top-1.")
    else:
        st.warning(f"⚠️ **Có sự khác biệt giữa 2 mô hình!** Xem so sánh chi tiết bên dưới.")

    # Render Models Side-by-Side
    if show_continual and res_cont:
        col_v2, col_v3, col_cont = st.columns(3)
    else:
        col_v2, col_v3 = st.columns(2)

    # Column 1: Finetune V2
    with col_v2:
        cfg = MODEL_CONFIGS["finetune_v2"]
        st.markdown(
            f"""
            <div class="answer-card" style="background-color: {cfg['bg_color']}; border-left: 5px solid {cfg['border_color']};">
                <div class="answer-title" style="color: {cfg['color']};">{cfg['title']} · TOP 1</div>
                <div class="answer-text" style="color: #1e293b;">{res_v2['top1']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Độ trễ (Latency)", f"{res_v2['latency_ms']:.2f} ms")
        with col_m2:
            st.metric("Tokens xuất", f"{len(res_v2['top1'].split())} từ")

        st.markdown(f"**Top {len(res_v2['hypotheses'])} Beam Candidates:**")
        for rank, (hyp, score) in enumerate(zip(res_v2["hypotheses"], res_v2.get("scores", [0]*len(res_v2["hypotheses"]))), 1):
            st.markdown(
                f'<div class="candidate-row"><span class="candidate-rank">{rank}</span><span class="candidate-text">{hyp}</span><span class="candidate-score">{score:.4f}</span></div>',
                unsafe_allow_html=True,
            )

    # Column 2: Base V3
    with col_v3:
        cfg = MODEL_CONFIGS["base_v3"]
        st.markdown(
            f"""
            <div class="answer-card" style="background-color: {cfg['bg_color']}; border-left: 5px solid {cfg['border_color']};">
                <div class="answer-title" style="color: {cfg['color']};">{cfg['title']} · TOP 1</div>
                <div class="answer-text" style="color: #1e293b;">{res_v3['top1']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Độ trễ (Latency)", f"{res_v3['latency_ms']:.2f} ms")
        with col_m2:
            st.metric("Tokens xuất", f"{len(res_v3['top1'].split())} từ")

        st.markdown(f"**Top {len(res_v3['hypotheses'])} Beam Candidates:**")
        for rank, (hyp, score) in enumerate(zip(res_v3["hypotheses"], res_v3.get("scores", [0]*len(res_v3["hypotheses"]))), 1):
            st.markdown(
                f'<div class="candidate-row"><span class="candidate-rank">{rank}</span><span class="candidate-text">{hyp}</span><span class="candidate-score">{score:.4f}</span></div>',
                unsafe_allow_html=True,
            )

    # Optional Column 3: Continual V1
    if show_continual and res_cont:
        with col_cont:
            cfg = MODEL_CONFIGS["continual_v1"]
            st.markdown(
                f"""
                <div class="answer-card" style="background-color: {cfg['bg_color']}; border-left: 5px solid {cfg['border_color']};">
                    <div class="answer-title" style="color: {cfg['color']};">{cfg['title']} · TOP 1</div>
                    <div class="answer-text" style="color: #1e293b;">{res_cont['top1']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Độ trễ (Latency)", f"{res_cont['latency_ms']:.2f} ms")
            with col_m2:
                st.metric("Tokens xuất", f"{len(res_cont['top1'].split())} từ")

            st.markdown(f"**Top {len(res_cont['hypotheses'])} Beam Candidates:**")
            for rank, (hyp, score) in enumerate(zip(res_cont["hypotheses"], res_cont.get("scores", [0]*len(res_cont["hypotheses"]))), 1):
                st.markdown(
                    f'<div class="candidate-row"><span class="candidate-rank">{rank}</span><span class="candidate-text">{hyp}</span><span class="candidate-score">{score:.4f}</span></div>',
                    unsafe_allow_html=True,
                )

    # Visual Diff Section if they differ
    if not is_agree:
        st.divider()
        st.markdown("### 🔍 So Sánh Sai Khác Từng Từ (Word-level Diff):")
        diff_v2, diff_v3 = highlight_diff(res_v2["top1"], res_v3["top1"])
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**Finetune V2:**")
            st.markdown(f'<div class="diff-box">{diff_v2}</div>', unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"**Base V3:**")
            st.markdown(f'<div class="diff-box">{diff_v3}</div>', unsafe_allow_html=True)

    # Feedback Section
    st.divider()
    fb_col1, fb_col2 = st.columns([1, 3])
    with fb_col1:
        report_btn = st.button("🚩 Ghi nhận ca lỗi / Feedback", use_container_width=True)
    with fb_col2:
        note_text = st.text_input("Ghi chú ca lỗi (tùy chọn):", placeholder="VD: V2 sai dấu ở chữ X, V3 sửa đúng...")

    if report_btn:
        saved, msg = save_feedback(query, res_v2, res_v3, note=note_text)
        st.success(msg)


if __name__ == "__main__":
    main()
