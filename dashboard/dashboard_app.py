import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.agent.llm_adapter import OllamaAdapter
from app.agent.state import SessionState
from app.pipeline import run_pipeline, run_pipeline_from_text
from app.stt.whisper_runner import WhisperRunner
from app.tts.piper_runner import PiperAdapter

st.set_page_config(page_title="Multilingual Voice Agent Benchmark", layout="wide")
st.title("Multilingual Voice Agent -- Benchmark Dashboard")

REPORTS_DIR = Path("benchmark/reports")

tab_analysis, tab_live = st.tabs(["Benchmark Analysis", "Try it Live"])


@st.cache_data
def load_reports() -> dict[str, dict]:
    reports = {}
    for path in sorted(REPORTS_DIR.glob("*.json")):
        with open(path, encoding="utf-8-sig") as f:
            reports[path.stem] = json.load(f)
    return reports


with tab_analysis:
    reports = load_reports()

    if not reports:
        st.warning("No benchmark reports found. Run `uv run python -m benchmark.runners.run_benchmark` first.")
        st.stop()

    selected = st.multiselect(
        "Select report(s) to compare",
        options=list(reports.keys()),
        default=list(reports.keys()),
    )

    if not selected:
        st.info("Select at least one report above.")
        st.stop()

    # --- Summary comparison table ---
    st.header("Summary")
    summary_rows = []
    for key in selected:
        r = reports[key]
        summary_rows.append({
            "Report": key,
            "Model": r.get("model_size", "unknown"),
            "Avg WER": r["avg_wer"],
            "Avg CER": r["avg_cer"],
            "Avg inference time (s)": r.get("avg_inference_time_s", None),
            "Language accuracy": r["language_accuracy"],
            "Code-switch F1": r["code_switch_f1"]["f1"],
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, width='stretch')

    # --- WER by language, per model (comparison chart) ---
    st.header("WER by language")
    lang_rows = []
    for key in selected:
        r = reports[key]
        for case in r["per_case"]:
            lang_rows.append({
                "Report": key,
                "Model": r.get("model_size", "unknown"),
                "Case": case["id"],
                "Language": case["reference_language"],
                "WER": case["wer"],
                "CER": case["cer"],
            })
    lang_df = pd.DataFrame(lang_rows)

    fig = px.bar(
        lang_df, x="Language", y="WER", color="Model", barmode="group",
        hover_data=["Case"], title="Word Error Rate by reference language",
    )
    st.plotly_chart(fig, width='stretch')

    # --- Latency comparison ---
    if "Avg inference time (s)" in summary_df.columns and summary_df["Avg inference time (s)"].notna().any():
        st.header("Inference latency")
        fig_latency = px.bar(summary_df, x="Report", y="Avg inference time (s)", color="Model")
        st.plotly_chart(fig_latency, width='stretch')

    # --- Failure analysis: worst cases ---
    st.header("Failure analysis -- worst cases by WER")
    worst_df = lang_df.sort_values("WER", ascending=False).head(10)
    st.dataframe(worst_df, width='stretch')

    st.header("Case detail -- reference vs hypothesis")
    for key in selected:
        r = reports[key]
        with st.expander(f"{key} (model: {r.get('model_size', 'unknown')})"):
            for case in r["per_case"]:
                st.markdown(f"**{case['id']}** (WER: {case['wer']:.3f}, CER: {case['cer']:.3f})")
                col1, col2 = st.columns(2)
                col1.text_area("Reference", case["reference_text"], height=80, key=f"{key}_{case['id']}_ref")
                col2.text_area("Hypothesis", case["hypothesis_text"], height=80, key=f"{key}_{case['id']}_hyp")

    if any("by_condition" in reports[key] for key in selected):
        st.header("Robustness by condition")
        cond_rows = []
        for key in selected:
            r = reports[key]
            if "by_condition" not in r:
                continue
            for cond, vals in r["by_condition"].items():
                cond_rows.append({
                    "Report": key,
                    "Model": r.get("model_size", "unknown"),
                    "Condition": cond,
                    "Avg WER": vals["avg_wer"],
                    "Avg CER": vals["avg_cer"],
                    "N cases": vals["num_cases"],
                })
        if cond_rows:
            cond_df = pd.DataFrame(cond_rows)
            fig_cond = px.line(
                cond_df, x="Condition", y="Avg WER", color="Model", markers=True,
                title="WER by robustness condition",
            )
            st.plotly_chart(fig_cond, width="stretch")
            st.dataframe(cond_df, width="stretch")


with tab_live:
    st.markdown("Talk to the agent directly -- upload a short audio clip, or type a message to skip STT.")

    @st.cache_resource
    def load_models():
        stt = WhisperRunner(model_size="small")
        llm = OllamaAdapter(model_name="llama3.2")
        tts = PiperAdapter()
        return stt, llm, tts

    if "live_session_id" not in st.session_state:
        st.session_state.live_session_id = str(uuid.uuid4())
        st.session_state.live_session = SessionState(session_id=st.session_state.live_session_id)
        st.session_state.live_turns = []

    col_reset, _ = st.columns([1, 5])
    if col_reset.button("New conversation"):
        st.session_state.live_session_id = str(uuid.uuid4())
        st.session_state.live_session = SessionState(session_id=st.session_state.live_session_id)
        st.session_state.live_turns = []
        st.rerun()

    mode = st.radio("Input mode", ["Type text", "Upload audio"], horizontal=True)

    if mode == "Type text":
        with st.form("text_form", clear_on_submit=True):
            user_text = st.text_input("Message (English, Urdu, or Roman Urdu)")
            submitted = st.form_submit_button("Send")
        if submitted and user_text.strip():
            with st.spinner("Loading models and generating response..."):
                stt, llm, tts = load_models()
                result = run_pipeline_from_text(user_text, st.session_state.live_session, llm, tts)
            st.session_state.live_turns.append(result)

    else:
        uploaded = st.file_uploader("Upload a short audio clip (wav/mp3)", type=["wav", "mp3"])
        if uploaded is not None and st.button("Send audio"):
            temp_path = f"data/audio/live_upload_{uuid.uuid4().hex[:8]}.wav" if uploaded.name.endswith(".wav") \
                else f"data/audio/live_upload_{uuid.uuid4().hex[:8]}.mp3"
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.spinner("Loading models and generating response..."):
                stt, llm, tts = load_models()
                result = run_pipeline(temp_path, st.session_state.live_session, stt, llm, tts)
            st.session_state.live_turns.append(result)

    st.divider()
    for turn in st.session_state.live_turns:
        with st.chat_message("user"):
            st.write(turn.transcript)
            st.caption(
                f"Detected: {turn.language_state.primary_language} "
                f"(script: {turn.language_state.script}, code-switch: {turn.language_state.code_switch_score})"
            )
        with st.chat_message("assistant"):
            st.write(turn.reply_text)
            st.audio(turn.tts_result.audio_path)
            st.caption(f"Voice: {turn.tts_result.voice_used} (native support: {turn.tts_result.native_support})")

