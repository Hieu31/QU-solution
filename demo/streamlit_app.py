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
        "note": "Curriculum-V2-Final · Vocab 8k · CTranslate2 beam 10",
        "ct2_dir": ROOT / "artifacts/reparos-curriculum-v2-final/ctranslate2",
        "tokenizer": ROOT / "artifacts/reparos-curriculum-v2-final/ctranslate2/tokenizer.model",
        "accent": "#f59e0b",
        "default_rep_penalty": 1.0,
    },
    "base_v3": {
        "title": "ReparoS Base V3",
        "note": "Production Clean · Vocab 12k · 6.5M params · beam 10",
        "ct2_dir": ROOT / "artifacts/checkpoints/base_v3_production/ctranslate2_export",
        "tokenizer": ROOT / "artifacts/checkpoints/base_v3_production/ctranslate2_export/tokenizer.model",
        "accent": "#2563eb",
        "default_rep_penalty": 1.2,
    },
    "continual_v1": {
        "title": "ReparoS Continual V1",
        "note": "Continual Fine-Tune Final · Vocab 8k · beam 10",
        "ct2_dir": ROOT / "artifacts/reparos-continual-v1-final/ctranslate2",
        "tokenizer": ROOT / "artifacts/reparos-continual-v1-final/ctranslate2/tokenizer.model",
        "accent": "#10b981",
        "default_rep_penalty": 1.0,
    },
}

MODEL_LOCK = threading.RLock()
FEEDBACK_LOCK = threading.RLock()
FEEDBACK_PATH = ROOT / "logs" / "demo_feedback.jsonl"


@st.cache_resource(show_spinner="Đang tải tokenizer & CTranslate2 model...")
def get_engine(ct2_dir: str, tokenizer_path: str, compute_type: str = "float32"):
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    translator = ctranslate2.Translator(
        str(ct2_dir),
        device="cpu",
        compute_type=compute_type,
        intra_threads=4,
        inter_threads=1,
    )
    # Warmup kernel and memory cache to eliminate cold-start latency spike
    warm_toks = sp.encode_as_pieces("ha noi")
    translator.translate_batch([warm_toks], beam_size=1)
    return sp, translator


def infer_single(
    sp: spm.SentencePieceProcessor,
    translator: ctranslate2.Translator,
    query: str,
    beam_size: int = 10,
    rep_penalty: float = 1.0,
    note: str = "",
) -> dict:
    if not query.strip():
        return {"top1": "", "hypotheses": [], "scores": [], "latency_ms": 0.0, "note": note}

    t0 = time.perf_counter_ns()
    tokens = sp.encode_as_pieces(query.strip().lower())

    with MODEL_LOCK:
        kwargs = {
            "beam_size": beam_size,
            "num_hypotheses": min(beam_size, 10),
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
        "note": note,
    }


def render_result(title: str, result: dict, accent: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(
        f'<div class="answer" style="border-color:{accent}"><span class="answer-label">TOP 1</span>{result["top1"]}</div>',
        unsafe_allow_html=True,
    )
    st.metric("Latency", f'{result["latency_ms"]:.2f} ms')
    st.caption(result.get("note", ""))
    st.markdown("**Top 10 (raw)**")
    scores = result.get("scores", [])
    for rank, hypothesis in enumerate(result["hypotheses"][:10], 1):
        score = f" · {scores[rank - 1]:.4f}" if rank <= len(scores) else ""
        st.markdown(
            f'<div class="candidate"><b>{rank}</b><span>{hypothesis}</span><small>{score}</small></div>',
            unsafe_allow_html=True,
        )


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
        page_title="ReparoS Lab: Finetune V2 vs Base V3",
        page_icon="⌕",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container { max-width: 1500px; padding-top: 1.8rem; }
        .answer {
            border-left: 5px solid;
            background: #f7f8fa;
            color: #111827;
            border-radius: 8px;
            padding: 14px 16px;
            font-size: 1.15rem;
            font-weight: 600;
            min-height: 76px;
        }
        .answer-label {
            display: block;
            color: #64748b;
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .12em;
            margin-bottom: 5px;
        }
        .candidate {
            display: grid;
            grid-template-columns: 24px 1fr auto;
            gap: 8px;
            padding: 7px 9px;
            border-bottom: 1px solid #e2e8f0;
            font-size: .89rem;
        }
        .candidate b { color: #94a3b8; }
        .candidate span { color: inherit; }
        .candidate small { color: #94a3b8; font-family: monospace; }
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

    st.title("Vietnamese Search Correction Lab")
    st.caption("So sánh trực tiếp **ReparoS Fine-Tune V2** · **ReparoS Base V3**")

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
        st.header("⚙️ Tuỳ chỉnh")
        beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=10, step=1, help="Beam 1 (Greedy): ~3ms. Beam 4: ~7ms. Beam 10: ~18ms")
        compute_type = st.selectbox(
            "Compute Type (CPU)",
            options=["float32", "int8", "int8_float32"],
            index=0,
            help="float32: Độ chính xác tuyệt đối. int8: Tối ưu hoá tập lệnh AVX2/AVX-512 CPU giúp giảm độ trễ 2x.",
        )
        v3_rep_penalty = st.slider(
            "Repetition Penalty (Base V3)",
            min_value=1.0,
            max_value=1.5,
            value=1.2,
            step=0.05,
            help="Hệ số phạt lặp từ cho Base V3 (khuyên dùng 1.2)",
        )
        show_continual = st.checkbox("Hiển thị thêm Continual V1 (3 cột)", value=False)
        debounce_ms = st.slider("Debounce khi gõ (ms)", min_value=100, max_value=1000, value=350, step=50)

        st.divider()
        st.markdown("### 📌 Thông số mô hình & Độ trễ:")
        st.markdown(
            """
            - **Finetune V2**:
              - Vocab: 8,000 subwords
              - FFN: 512
              - Latency: ~10-16ms (Beam 10)
            - **Base V3**:
              - Vocab: 12,000 subwords (+50% matrix size)
              - FFN: 2048 (4x dung lượng FFN biểu diễn)
              - Latency Beam 1: **~3ms** (SLA production)
              - Latency Beam 4: **~7-10ms**
              - Latency Beam 10: **~18-40ms** (FP32 CPU)
            """
        )

    # Helper for preset button clicks
    if "search_counter" not in st.session_state:
        st.session_state["search_counter"] = 0
    if "last_applied_counter" not in st.session_state:
        st.session_state["last_applied_counter"] = -1
    if "preset_query" not in st.session_state:
        st.session_state["preset_query"] = "cau vuot song than"

    def select_preset(text: str):
        st.session_state["preset_query"] = text
        st.session_state["search_counter"] += 1

    # Preset queries for quick testing from real Zero-Click logs
    st.markdown("##### 📂 Ca mẫu thực tế trích xuất từ `zero_click.csv` (Click để test ngay):")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏙️ Địa danh & Chợ & Nút giao",
        "🏥 Bệnh viện & Trường học & TTTM",
        "📍 Địa chỉ & Viết tắt hành chính",
        "⚠️ Gõ dở dang & Vướng phím Telex",
    ])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button("🌉 cau vuot song than", on_click=select_preset, args=("cau vuot song than",), use_container_width=True, help="V2: sông than (sai) | V3: sóng thần (đúng)")
        with c2:
            st.button("🛒 cho ba chieu", on_click=select_preset, args=("cho ba chieu",), use_container_width=True, help="V2: ba chiêu (sai) | V3: bà chiểu (đúng)")
        with c3:
            st.button("🚦 nga 4 hang xanh", on_click=select_preset, args=("nga 4 hang xanh",), use_container_width=True)
        with c4:
            st.button("⚓ nga 3 vung tau", on_click=select_preset, args=("nga 3 vung tau",), use_container_width=True)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button("🏥 bv cho ray", on_click=select_preset, args=("bv cho ray",), use_container_width=True, help="V2: chợ ray (sai) | V3: chợ rẫy (đúng)")
        with c2:
            st.button("👶 Bv nhi đong", on_click=select_preset, args=("Bv nhi đong",), use_container_width=True)
        with c3:
            st.button("🚑 benh vien 175", on_click=select_preset, args=("benh vien 175",), use_container_width=True)
        with c4:
            st.button("🎓 dh kinh te tphcm", on_click=select_preset, args=("dh kinh te tphcm",), use_container_width=True)

    with tab3:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button("🏠 86 xo viet nghe tinh p19 binh thanh", on_click=select_preset, args=("86 xo viet nghe tinh p19 binh thanh",), use_container_width=True)
        with c2:
            st.button("🛣️ duong le van viet q9", on_click=select_preset, args=("duong le van viet q9",), use_container_width=True)
        with c3:
            st.button("🚪 hem 212 thoai ngoc hau phuong phu thanh", on_click=select_preset, args=("hem 212 thoai ngoc hau phuong phu thanh",), use_container_width=True)
        with c4:
            st.button("🏭 kcn song than 1", on_click=select_preset, args=("kcn song than 1",), use_container_width=True)

    with tab4:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button("⌨️ 158/16 binh quew", on_click=select_preset, args=("158/16 binh quew",), use_container_width=True, help="V2: bình quế (bịa từ) | V3: bình quêw (bảo toàn)")
        with c2:
            st.button("🏢 chung cu ha", on_click=select_preset, args=("chung cu ha",), use_container_width=True, help="Gõ dở dang: V2 đoán bừa 'hạ' | V3 giữ 'ha'")
        with c3:
            st.button("🔤 ngã 6 tahnhf", on_click=select_preset, args=("ngã 6 tahnhf",), use_container_width=True, help="Lỗi gõ phím đảo: tahnhf")
        with c4:
            st.button("🏬 tttm aeon mall tan phu", on_click=select_preset, args=("tttm aeon mall tan phu",), use_container_width=True)

    # If user clicked a preset, pass it once on mount of the new key; otherwise pass "" so JS does not rubberband when deleted!
    is_new_preset = (st.session_state["search_counter"] != st.session_state["last_applied_counter"])
    if is_new_preset:
        initial_box_val = st.session_state["preset_query"]
        st.session_state["last_applied_counter"] = st.session_state["search_counter"]
    else:
        initial_box_val = ""

    active_key = f"live_search_input_{st.session_state['search_counter']}"
    raw_query = st_keyup(
        "Nhập hoặc chọn truy vấn địa điểm cần sửa lỗi:",
        value=initial_box_val,
        debounce=debounce_ms,
        key=active_key,
    )
    query = (raw_query or "").strip()

    if not query:
        st.info("Bắt đầu nhập để chạy so sánh."); return

    # Load engines
    sp_v2, trans_v2 = get_engine(str(MODEL_CONFIGS["finetune_v2"]["ct2_dir"]), str(MODEL_CONFIGS["finetune_v2"]["tokenizer"]), compute_type=compute_type)
    sp_v3, trans_v3 = get_engine(str(MODEL_CONFIGS["base_v3"]["ct2_dir"]), str(MODEL_CONFIGS["base_v3"]["tokenizer"]), compute_type=compute_type)

    with st.spinner("Đang sửa truy vấn…"):
        res_v2 = infer_single(
            sp_v2, trans_v2, query,
            beam_size=beam_size, rep_penalty=1.0,
            note=f"CTranslate2 · beam {beam_size} · {compute_type}"
        )
        res_v3 = infer_single(
            sp_v3, trans_v3, query,
            beam_size=beam_size, rep_penalty=v3_rep_penalty,
            note=f"CTranslate2 · beam {beam_size} · {compute_type} · rep_pen {v3_rep_penalty}"
        )

        res_cont = None
        if show_continual and MODEL_CONFIGS["continual_v1"]["ct2_dir"].exists():
            sp_cont, trans_cont = get_engine(str(MODEL_CONFIGS["continual_v1"]["ct2_dir"]), str(MODEL_CONFIGS["continual_v1"]["tokenizer"]))
            res_cont = infer_single(
                sp_cont, trans_cont, query,
                beam_size=beam_size, rep_penalty=1.0,
                note=f"CTranslate2 · beam {beam_size}"
            )

    # Agreement / Disagreement Banner
    is_agree = (res_v2["top1"].strip().lower() == res_v3["top1"].strip().lower())
    if is_agree:
        st.success("✅ **Đồng thuận 100%**: Cả Finetune V2 và Base V3 đều trả về cùng kết quả Top 1.")
    else:
        st.warning("⚠️ **Có sự khác biệt giữa 2 mô hình!**")

    # Render Models Side-by-Side in Columns
    if show_continual and res_cont:
        col_v2, col_v3, col_cont = st.columns(3)
        with col_v2:
            render_result(MODEL_CONFIGS["finetune_v2"]["title"], res_v2, MODEL_CONFIGS["finetune_v2"]["accent"])
        with col_v3:
            render_result(MODEL_CONFIGS["base_v3"]["title"], res_v3, MODEL_CONFIGS["base_v3"]["accent"])
        with col_cont:
            render_result(MODEL_CONFIGS["continual_v1"]["title"], res_cont, MODEL_CONFIGS["continual_v1"]["accent"])
    else:
        col_v2, col_v3 = st.columns(2)
        with col_v2:
            render_result(MODEL_CONFIGS["finetune_v2"]["title"], res_v2, MODEL_CONFIGS["finetune_v2"]["accent"])
        with col_v3:
            render_result(MODEL_CONFIGS["base_v3"]["title"], res_v3, MODEL_CONFIGS["base_v3"]["accent"])

    # Visual Diff Section if they differ
    if not is_agree:
        st.divider()
        st.markdown("### 🔍 So sánh khác biệt từ ngữ (Diff):")
        diff_v2, diff_v3 = highlight_diff(res_v2["top1"], res_v3["top1"])
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**{MODEL_CONFIGS['finetune_v2']['title']}:**")
            st.markdown(f'<div class="diff-box">{diff_v2}</div>', unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"**{MODEL_CONFIGS['base_v3']['title']}:**")
            st.markdown(f'<div class="diff-box">{diff_v3}</div>', unsafe_allow_html=True)

    # Feedback Section
    st.divider()
    fb_col1, fb_col2 = st.columns([1, 4])
    with fb_col1:
        report_btn = st.button("🚩 Báo kết quả sai", type="primary", use_container_width=True)
    with fb_col2:
        note_text = st.text_input("Ghi chú (tuỳ chọn):", placeholder="Nhận xét của bạn về ca này...")

    if report_btn:
        saved, msg = save_feedback(query, res_v2, res_v3, note=note_text)
        st.success(msg)


if __name__ == "__main__":
    main()
