"""
streamlit_app.py
----------------
Streamlit frontend for the Crop Recommendation model.

Run with:
    streamlit run streamlit_app.py

"""

import json
import os
import streamlit as st
from src.predict import predict_crop

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="centered",
)

# ── Crop emoji map ────────────────────────────────────────────────────────────
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍑", "coconut": "🥥", "cotton": "🌿",
    "jute": "🌿", "coffee": "☕", "wheat": "🌾", "tobacco": "🌿",
}


def get_emoji(crop: str) -> str:
    return CROP_EMOJI.get(crop.lower(), "🌱")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 Crop Recommendation")
st.markdown(
    "Enter your **soil and climate conditions** below "
    "and get an instant crop recommendation."
)
st.divider()

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")

    metrics_path = os.path.join("models", "metrics.json")
    labels_path  = os.path.join("models", "label_classes.json")

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
        st.metric("F1 Score",  f"{metrics['f1']*100:.1f}%")
        st.metric("Precision", f"{metrics['precision']*100:.1f}%")
        st.metric("Recall",    f"{metrics['recall']*100:.1f}%")
    else:
        st.warning("Model not trained yet.\nRun: `python -m src.train`")

    if os.path.exists(labels_path):
        with open(labels_path) as f:
            classes = json.load(f)
        st.divider()
        st.markdown(f"**{len(classes)} crops supported:**")
        for c in classes:
            st.markdown(f"{get_emoji(c)} {c.capitalize()}")


# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("🧪 Soil Conditions")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input(
        "Nitrogen (N)", min_value=0.0, max_value=140.0,
        value=90.0, step=1.0, help="Nitrogen content in soil (kg/ha)"
    )

with col2:
    P = st.number_input(
        "Phosphorus (P)", min_value=5.0, max_value=145.0,
        value=42.0, step=1.0, help="Phosphorus content in soil (kg/ha)"
    )

with col3:
    K = st.number_input(
        "Potassium (K)", min_value=5.0, max_value=205.0,
        value=43.0, step=1.0, help="Potassium content in soil (kg/ha)"
    )

ph = st.slider(
    "Soil pH", min_value=3.5, max_value=10.0,
    value=6.5, step=0.1,
    help="pH value of the soil (7 = neutral, <7 = acidic, >7 = alkaline)"
)

# pH indicator
if ph < 6.0:
    st.caption("⚠️ Acidic soil")
elif ph > 7.5:
    st.caption("⚠️ Alkaline soil")
else:
    st.caption("✅ Neutral — good for most crops")

st.divider()
st.subheader("🌤️ Climate Conditions")

col4, col5 = st.columns(2)

with col4:
    temperature = st.slider(
        "Temperature (°C)", min_value=8.0, max_value=44.0,
        value=20.8, step=0.1
    )
    humidity = st.slider(
        "Humidity (%)", min_value=14.0, max_value=100.0,
        value=82.0, step=0.5
    )

with col5:
    rainfall = st.slider(
        "Rainfall (mm)", min_value=20.0, max_value=300.0,
        value=202.9, step=1.0
    )

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🌱 Get Recommendation", type="primary", use_container_width=True):

    if not os.path.exists("models/crop_model.pkl"):
        st.error("❌ Model not found. Run training first: `python -m src.train`")
    else:
        with st.spinner("Analyzing conditions..."):
            try:
                result = predict_crop(
                    N=N, P=P, K=K,
                    temperature=temperature,
                    humidity=humidity,
                    ph=ph,
                    rainfall=rainfall,
                )

                crop       = result["crop"]
                confidence = result["confidence"]
                emoji      = get_emoji(crop)

                # ── Result card ───────────────────────────────────────────────
                st.success("✅ Recommendation ready!")

                st.markdown(
                    f"""
                    <div style="
                        background: #f0fdf4;
                        border: 1.5px solid #86efac;
                        border-radius: 12px;
                        padding: 24px 32px;
                        text-align: center;
                        margin: 16px 0;
                    ">
                        <div style="font-size: 56px;">{emoji}</div>
                        <div style="font-size: 32px; font-weight: 600;
                                    color: #166534; margin: 8px 0;">
                            {crop.capitalize()}
                        </div>
                        <div style="font-size: 15px; color: #4b5563;">
                            Model confidence: <strong>{confidence*100:.0f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Confidence bar ────────────────────────────────────────────
                st.markdown("**Confidence level**")
                st.progress(confidence)

                if confidence >= 0.80:
                    st.caption("🟢 High confidence — strong match for your conditions")
                elif confidence >= 0.50:
                    st.caption("🟡 Medium confidence — suitable but not ideal conditions")
                else:
                    st.caption("🔴 Low confidence — consider adjusting soil conditions")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Model: Naive Bayes · Dataset: Crop Recommendation (Kaggle)")