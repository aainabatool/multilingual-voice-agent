import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Multilingual Voice Agent Benchmark", layout="wide")
st.title("Multilingual Voice Agent -- Benchmark Dashboard")

REPORTS_DIR = Path("benchmark/reports")


@st.cache_data
def load_reports() -> dict[str, dict]:
    reports = {}
    for path in sorted(REPORTS_DIR.glob("*.json")):
        with open(path, encoding="utf-8-sig") as f:
            reports[path.stem] = json.load(f)
    return reports


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
