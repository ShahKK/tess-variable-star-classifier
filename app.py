from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import pandas as pd
import streamlit as st

from tess_classifier.config import DATASET_SUMMARY_FILE, METRICS_FILE, MODEL_FILE, TESS_CURATED_DIR
from tess_classifier.data import (
    LABELS,
    generate_synthetic_light_curve,
    list_local_light_curve_files,
    load_existing_manifest,
    load_light_curve_csv,
)
from tess_classifier.features import extract_features


st.set_page_config(page_title="TESS Variable Star Classifier", page_icon="stars")
st.title("TESS Variable Star Classifier")
st.caption("A small AI + astrophysics demo for stellar variability classification.")

if not MODEL_FILE.exists():
    st.warning("No trained model found yet. Run `python train.py` first, then refresh this page.")
    st.stop()

artifact = joblib.load(MODEL_FILE)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
manifest = load_existing_manifest()
real_manifest = pd.DataFrame()
if not manifest.empty and "is_real_data" in manifest.columns:
    real_manifest = manifest[pd.Series(manifest["is_real_data"]).fillna(False).astype(bool)].copy()

if not real_manifest.empty:
    local_curve_files = [
        TESS_CURATED_DIR / Path(relative_path)
        for relative_path in real_manifest["relative_path"].astype(str).tolist()
    ]
else:
    local_curve_files = list_local_light_curve_files(TESS_CURATED_DIR)

with st.sidebar:
    st.header("Demo Controls")
    input_modes = ["synthetic"]
    if local_curve_files:
        input_modes.append("local_tess")
    input_source = st.radio("Input source", input_modes)

    selected_label = None
    selected_curve = None
    seed = None
    if input_source == "synthetic":
        selected_label = st.selectbox("Synthetic archetype", LABELS)
        seed = st.number_input("Random seed", min_value=0, value=7, step=1)
    else:
        display_options = {str(path.relative_to(TESS_CURATED_DIR)): path for path in local_curve_files}
        selected_key = st.selectbox("Curated light curve", list(display_options.keys()))
        selected_curve = display_options[selected_key]

if input_source == "synthetic":
    time_axis, flux = generate_synthetic_light_curve(selected_label, seed=int(seed))
    source_label = selected_label
    source_path = "synthetic"
    source_note = "Synthetic demo light curve"
else:
    time_axis, flux = load_light_curve_csv(selected_curve)
    source_label = selected_curve.parent.name
    source_path = str(selected_curve.relative_to(TESS_CURATED_DIR)).replace("\\", "/")
    source_note = ""
    if not real_manifest.empty:
        matched_rows = real_manifest[real_manifest["relative_path"] == source_path]
        if not matched_rows.empty:
            source_note = str(matched_rows.iloc[0].get("notes", ""))

features = extract_features(time_axis, flux)
feature_frame = pd.DataFrame([[features[name] for name in feature_columns]], columns=feature_columns)

prediction = model.predict(feature_frame)[0]
probabilities = model.predict_proba(feature_frame)[0]
proba_frame = pd.DataFrame(
    {"class": model.classes_, "probability": probabilities}
).sort_values("probability", ascending=False)

st.subheader("Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Input source", input_source)
col2.metric("Expected label", source_label)
col3.metric("Model prediction", prediction)
st.caption(f"Source file: {source_path}")
if source_note:
    st.caption(source_note)

st.subheader("Light Curve")
chart_frame = pd.DataFrame({"time": time_axis, "flux": flux}).set_index("time")
st.line_chart(chart_frame)

st.subheader("Class Probabilities")
st.dataframe(proba_frame, use_container_width=True, hide_index=True)

st.subheader("Extracted Features")
feature_display = pd.DataFrame(
    {"feature": list(features.keys()), "value": [round(value, 6) for value in features.values()]}
)
st.dataframe(feature_display, use_container_width=True, hide_index=True)

if METRICS_FILE.exists():
    st.subheader("Latest Training Metrics")
    st.json(json.loads(METRICS_FILE.read_text(encoding="utf-8")))
if DATASET_SUMMARY_FILE.exists():
    st.subheader("Curated Dataset Summary")
    summary = json.loads(DATASET_SUMMARY_FILE.read_text(encoding="utf-8"))
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total curated files", summary.get("num_files", 0))
    metric_col2.metric("Official real files", summary.get("real_data_files", 0))
    metric_col3.metric("Distinct labels", summary.get("num_labels", 0))
    st.json(summary)
    if summary.get("real_data_files", 0) > 0:
        st.info(
            "Training currently uses the real-data entries marked in the manifest. "
            "The synthetic placeholder CSVs remain in the folder for format reference."
        )
elif not local_curve_files:
    st.info(
        "Add labeled CSV files under data/tess_curated/<label>/sample.csv to inspect real light curves "
        "in this app."
    )
