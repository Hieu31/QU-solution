from __future__ import annotations

import sys
import threading
import time
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st
from st_keyup import st_keyup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarking.io import normalize_text  # noqa: E402
from benchmarking.runners import load_decoding  # noqa: E402
from benchmarking.smoke_latency import build_opennmt_engine  # noqa: E402
from reparos.serving.ctranslate2 import CTranslate2Predictor  # noqa: E402
from webspell.pipeline.model import WebSpellModel  # noqa: E402

DEFAULTS = {
    "webspell": ROOT / "artifacts/osm-model-production-v4-leakfree",
    "checkpoint": ROOT / "artifacts/opennmt-production-kaggle-t4-v1/reparos_base_step_25000.pt",
    "tokenizer": ROOT / "artifacts/tokenizer-production-v1/tokenizer.model",
    "decoding": ROOT / "artifacts/opennmt-production-kaggle-t4-v1/decoding-config.json",
    "ctranslate2": ROOT / "artifacts/opennmt-production-kaggle-t4-v1-ctranslate2-float32",
}
WEBSPELL_LOCK = threading.RLock()
OPENNMT_LOCK = threading.RLock()
CTRANSLATE2_LOCK = threading.RLock()
FEEDBACK_LOCK = threading.RLock()
FEEDBACK_PATH = ROOT / "logs" / "demo_feedback.jsonl"


@st.cache_resource(show_spinner="Đang tải WebSpell…")
def load_webspell(path: str):
    return WebSpellModel.load(path)


@st.cache_resource(show_spinner="Đang tải OpenNMT reference…")
def load_opennmt(checkpoint: str, tokenizer: str, decoding: str):
    return build_opennmt_engine(checkpoint, tokenizer, decoding)


@st.cache_resource(show_spinner="Đang tải CTranslate2…")
def load_ctranslate2(path: str):
    return CTranslate2Predictor(path, device="cpu", compute_type="float32")


def webspell_alternatives(model, query: str) -> dict:
    started = time.perf_counter_ns()
    with WEBSPELL_LOCK:
        predictions = model.predict(query)
        top1 = normalize_text(model.corrected_text(query, predictions))
    alternatives = [top1]
    for prediction in predictions:
        token = prediction.token
        for candidate in prediction.candidates:
            text = normalize_text(query[:token.character_start] + candidate.term + query[token.character_end:])
            if text not in alternatives:
                alternatives.append(text)
            if len(alternatives) == 10:
                break
        if len(alternatives) == 10:
            break
    return {"top1": top1, "hypotheses": alternatives, "scores": [],
            "latency_ms": (time.perf_counter_ns() - started) / 1_000_000,
            "note": "Token candidates ghép thành query; không phải sequence beam."}


def opennmt_predict(engine, query: str) -> dict:
    started = time.perf_counter_ns()
    # OpenNMT's translator owns mutable decoder/beam caches. A cached engine
    # cannot safely serve overlapping Streamlit session threads.
    with OPENNMT_LOCK:
        scores, predictions = engine.infer_list([query])
    hypotheses = [normalize_text(str(value)) for value in predictions[0]]
    return {"top1": hypotheses[0], "hypotheses": hypotheses,
            "scores": [float(value) for value in scores[0]],
            "latency_ms": (time.perf_counter_ns() - started) / 1_000_000,
            "note": "OpenNMT-py reference · beam 10"}


def ct2_predict(model, query: str, decoding_path: str) -> dict:
    with CTRANSLATE2_LOCK:
        result = model.predict(query, decoding=load_decoding(decoding_path))
    return {"top1": normalize_text(str(result["top1_query"])),
            "hypotheses": [normalize_text(str(value)) for value in result["hypotheses"]],
            "scores": [float(value) for value in result["sequence_scores"]],
            "latency_ms": float(result["latency_ms"]),
            "note": "CTranslate2 production · beam 10"}


def render_result(title: str, result: dict, accent: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(f'<div class="answer" style="border-color:{accent}"><span class="answer-label">TOP 1</span>{result["top1"]}</div>', unsafe_allow_html=True)
    st.metric("Latency", f'{result["latency_ms"]:.2f} ms')
    st.caption(result["note"])
    st.markdown("**Top 10 (raw)**")
    scores = result.get("scores", [])
    for rank, hypothesis in enumerate(result["hypotheses"][:10], 1):
        score = f" · {scores[rank - 1]:.4f}" if rank <= len(scores) else ""
        st.markdown(f'<div class="candidate"><b>{rank}</b><span>{hypothesis}</span><small>{score}</small></div>', unsafe_allow_html=True)


def save_feedback(query: str, results: tuple[dict, dict, dict]) -> tuple[bool, str]:
    systems = {}
    for name, result in zip(("webspell", "opennmt", "ctranslate2"), results, strict=True):
        systems[name] = {
            "top1": result["top1"],
            "hypotheses": list(result["hypotheses"][:10]),
            "scores": list(result.get("scores", [])[:10]),
            "latency_ms": float(result["latency_ms"]),
        }
    fingerprint = hashlib.sha256(
        json.dumps({
            "input": query,
            "outputs": {name: {"top1": value["top1"], "hypotheses": value["hypotheses"]}
                        for name, value in systems.items()},
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if st.session_state.get("last_feedback_fingerprint") == fingerprint:
        return False, "Kết quả này đã được gửi trong phiên hiện tại."
    record = {
        "schema_version": 1,
        "feedback_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": query,
        "systems": systems,
    }
    with FEEDBACK_LOCK:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    st.session_state["last_feedback_fingerprint"] = fingerprint
    return True, f"Đã lưu phản hồi · {record['feedback_id'][:8]}"


def main() -> None:
    st.set_page_config(page_title="Vietnamese Search Correction Lab", page_icon="⌕", layout="wide")
    st.markdown("""<style>
    .block-container{max-width:1500px;padding-top:2rem}.answer{border-left:5px solid;background:#f7f8fa;color:#111827;border-radius:8px;padding:14px 16px;font-size:1.15rem;font-weight:600;min-height:76px}.answer-label{display:block;color:#64748b;font-size:.68rem;font-weight:700;letter-spacing:.12em;margin-bottom:5px}.candidate{display:grid;grid-template-columns:24px 1fr auto;gap:8px;padding:7px 9px;border-bottom:1px solid #374151;font-size:.89rem}.candidate b{color:#94a3b8}.candidate span{color:inherit}.candidate small{color:#94a3b8}
    </style>""", unsafe_allow_html=True)
    st.title("Vietnamese Search Correction Lab")
    st.caption("So sánh trực tiếp WebSpell · OpenNMT-py · CTranslate2")
    paths = {key: str(value) for key, value in DEFAULTS.items()}
    missing = [key for key, value in paths.items() if not Path(value).exists()]
    if missing:
        st.error("Thiếu artifact: " + ", ".join(missing)); st.stop()
    with st.expander("Tuỳ chỉnh", expanded=False):
        debounce = st.slider("Debounce khi gõ (ms)", 100, 1000, 350, 50)
    query = normalize_text(st_keyup("Nhập truy vấn địa điểm", value="san bay noi bai", debounce=debounce, key="live_query") or "")
    if not query:
        st.info("Bắt đầu nhập để chạy cả ba phương án."); return
    webspell = load_webspell(paths["webspell"])
    opennmt = load_opennmt(paths["checkpoint"], paths["tokenizer"], paths["decoding"])
    ctranslate2 = load_ctranslate2(paths["ctranslate2"])
    with st.spinner("Đang sửa truy vấn…"):
        results = (
            webspell_alternatives(webspell, query),
            opennmt_predict(opennmt, query),
            ct2_predict(ctranslate2, query, paths["decoding"]),
        )
    for column, title, result, accent in zip(st.columns(3), ("WebSpell", "OpenNMT", "CTranslate2"), results, ("#0f766e", "#7c3aed", "#2563eb"), strict=True):
        with column:
            render_result(title, result, accent)
    st.divider()
    feedback_col, message_col = st.columns([1, 4])
    with feedback_col:
        report = st.button("🚩 Báo kết quả sai", type="primary", use_container_width=True)
    if report:
        saved, message = save_feedback(query, results)
        with message_col:
            (st.success if saved else st.info)(message)


if __name__ == "__main__":
    main()
