# streamlit_sales_xgb.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from io import BytesIO

st.set_page_config(page_title="Sales Forecast (XGBoost)", layout="centered")

MODEL_DIR = "saved_sales_model"

@st.cache_data
def load_metadata(path=os.path.join(MODEL_DIR, "metadata.json")):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_model(model_file):
    return joblib.load(os.path.join(MODEL_DIR, model_file))

# Load model and metadata
try:
    metadata = load_metadata()
    model = load_model(metadata["model_file"])
    FEATURE_NAMES = metadata["feature_names"]
    LAG = int(metadata["lag"])
except Exception as e:
    st.error(f"Could not load model/metadata from '{MODEL_DIR}'. Make sure 'metadata.json' and model joblib are present. Error: {e}")
    st.stop()

# Title and info
st.title("📈 Sales Forecast Predictor (XGBoost)")
st.markdown(f"Model expects **{LAG} lag features**: {', '.join(FEATURE_NAMES)}")

# Use tabs instead of sidebar for mode selection
tab1, tab2 = st.tabs(["🔹 Single Input", "📂 Batch CSV"])

# -------- SINGLE INPUT TAB --------
with tab1:
    st.subheader("Single Prediction")
    st.markdown("Paste your last historical sales values (oldest → newest).")

    text = st.text_area(
        f"Enter at least {LAG} historical sales values (comma-separated):",
        height=120,
        placeholder="e.g. 120, 135, 142, 155, 160, 178 ..."
    )

    def parse_comma_series(text):
        try:
            parts = [p.strip() for p in text.split(",") if p.strip() != ""]
            arr = np.array([float(p) for p in parts])
            return arr
        except Exception as e:
            st.error(f"Could not parse input series: {e}")
            return None

    def build_feature_vector_from_series(series_arr):
        if len(series_arr) < LAG:
            st.error(f"Need at least {LAG} historical sales values. Provided: {len(series_arr)}")
            return None
        last_vals = series_arr[-LAG:]
        feat = [float(last_vals[-1 - i]) for i in range(LAG)]
        return np.array(feat).reshape(1, -1)

    if text:
        series = parse_comma_series(text)
        if series is not None:
            st.write("Preview (last 10 values):", list(series[-10:]))
            feat_vec = build_feature_vector_from_series(series)
            if feat_vec is not None:
                if st.button("🔮 Predict Next Sales Value"):
                    try:
                        pred = model.predict(feat_vec)[0]
                        st.metric("Predicted Next Sales", f"{pred:.4f}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

# -------- BATCH PREDICTION TAB --------
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.markdown(f"Your CSV must include these columns: `{', '.join(FEATURE_NAMES)}`")

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    def batch_predict_from_dataframe(df):
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
            return None
        X = df[FEATURE_NAMES].astype(float).values
        preds = model.predict(X)
        out = df.copy()
        out["prediction"] = preds
        return out

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Uploaded CSV Preview:")
            st.dataframe(df.head(5))

            out_df = batch_predict_from_dataframe(df)
            if out_df is not None:
                st.success(f"✅ Predicted {len(out_df)} rows.")
                st.dataframe(out_df.head(50))

                towrite = BytesIO()
                out_df.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button(
                    "💾 Download Predictions CSV",
                    data=towrite,
                    file_name="sales_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# -------- FOOTER --------
st.markdown("---")
st.caption("Model loaded from folder 'saved_sales_model' (sales_xgb.joblib + metadata.json).")
