"""Streamlit demo: text box, live scores, decision badge, latency. Phase 7.

Loads the ONNX-INT8 model in-process via st.cache_resource — Streamlit's
equivalent of the "load once" singleton pattern src/serving/app.py's FastAPI
lifespan handler uses (Architecture.md Decision #1) — rather than calling a
separately-running API server, so `streamlit run demo/streamlit_app.py`
works standalone per Phases.md's Phase 7 acceptance criterion.

Preset examples are deliberately mild (per Design.md: "use mild examples for
the 'toxic' preset") — reviewers get one-click examples without this file
ever containing real toxic/offensive language (Rules.md).
"""

import json
from pathlib import Path

import streamlit as st

from src.data.preprocess import load_label_names, load_training_config
from src.models.export_onnx import QUANTIZED_FILE_NAME
from src.serving.inference import ModerationModel, decide, label_band

MODEL_DIR = Path("models/onnx/int8")
THRESHOLDS_PATH = Path("configs/thresholds.json")

BAND_COLOR = {"allow": "#22C55E", "flag": "#F59E0B", "block": "#EF4444"}

PRESET_MILD = "Thanks so much for the quick response, really appreciate the help!"
PRESET_TOXIC_LEANING = "You're being kind of annoying honestly, just stop already."
PRESET_OBFUSCATED = "u are s0 dumb lol, just st0p already"


@st.cache_resource
def load_resources():
    label_names = load_label_names()
    config = load_training_config()
    model = ModerationModel(MODEL_DIR, QUANTIZED_FILE_NAME, config["model"]["max_length"], label_names)
    with THRESHOLDS_PATH.open() as f:
        thresholds = json.load(f)
    return model, thresholds, label_names


def run_check(text: str, model: ModerationModel, thresholds: dict) -> None:
    if not text.strip():
        st.session_state.result = None
        return
    scores, latency_ms = model.predict(text)
    st.session_state.result = {"scores": scores, "decision": decide(scores, thresholds), "latency_ms": latency_ms}


st.set_page_config(page_title="ModGuard", page_icon="🛡", layout="centered")

if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "result" not in st.session_state:
    st.session_state.result = None

st.title("🛡 ModGuard — real-time toxicity check")
st.caption("Fine-tuned DistilBERT, ONNX Runtime INT8")

try:
    model, thresholds, label_names = load_resources()
except Exception as exc:
    st.error(
        "Model files not found at `models/onnx/int8/` — this repo doesn't commit model "
        "weights (see .gitignore). Follow the README's quickstart to produce or download "
        f"them before running the demo.\n\nDetail: {exc}"
    )
    st.stop()


def set_preset(text: str) -> None:
    st.session_state.text_input = text
    run_check(text, model, thresholds)


st.text_area("Comment", key="text_input", placeholder="Type a comment...", height=100, label_visibility="collapsed")

col1, col2, col3, col4 = st.columns(4)
check_clicked = col1.button("Check comment", type="primary")
col2.button("Preset: mild", on_click=set_preset, args=(PRESET_MILD,))
col3.button("Preset: toxic-leaning", on_click=set_preset, args=(PRESET_TOXIC_LEANING,))
col4.button("Preset: obfuscated", on_click=set_preset, args=(PRESET_OBFUSCATED,))

if check_clicked:
    run_check(st.session_state.text_input, model, thresholds)

result = st.session_state.result
if result:
    badge_color = BAND_COLOR[result["decision"]]
    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div style="background:{badge_color};color:#0F172A;font-weight:700;'
            f'padding:10px 16px;border-radius:8px;text-align:center;font-size:1.1rem;">'
            f'{result["decision"].upper()}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.code(f"Checked in {result['latency_ms']:.1f} ms on CPU · model: ONNX INT8", language=None)

    st.subheader("Per-label scores")
    for label in sorted(label_names, key=lambda l: result["scores"][l], reverse=True):
        score = result["scores"][label]
        label_thresholds = thresholds["labels"][label]
        color = BAND_COLOR[label_band(score, label_thresholds)]
        bar_pct = score * 100
        flag_pct = label_thresholds["flag_threshold"] * 100
        block_pct = label_thresholds["block_threshold"] * 100
        st.markdown(
            f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#94A3B8;">
                <span>{label}</span><span style="font-family:monospace;">{score:.3f}</span>
              </div>
              <div style="position:relative;background:#1E293B;border-radius:4px;height:10px;">
                <div style="position:absolute;left:0;top:0;bottom:0;width:{bar_pct:.1f}%;
                            background:{color};border-radius:4px;"></div>
                <div style="position:absolute;left:{flag_pct:.1f}%;top:-2px;bottom:-2px;
                            width:2px;background:#E2E8F0;opacity:0.5;"></div>
                <div style="position:absolute;left:{block_pct:.1f}%;top:-2px;bottom:-2px;
                            width:2px;background:#E2E8F0;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("How this works"):
    st.markdown(
        "Text → DistilBERT tokenizer → fine-tuned DistilBERT (multi-label, sigmoid per "
        "label) → exported to ONNX and INT8-quantized for CPU speed → per-label scores "
        "compared against thresholds tuned from precision-recall curves on held-out data "
        "(`configs/thresholds.json`) → allow / flag / block decision. Tick marks on each "
        "bar show that label's flag and block thresholds.\n\n"
        "See the [repo](https://github.com/AniketPatil1710/real-time-content-moderation) "
        "for the full training and evaluation pipeline."
    )

st.caption(
    "Scores from DistilBERT fine-tuned on Jigsaw Toxic Comment + Civil Comments data · "
    "thresholds tuned for ≥90% precision at the block tier"
)
st.caption("Demo system trained on public datasets; not a production moderation service.")
