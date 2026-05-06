
import streamlit as st
import pickle
import numpy as np
import time

# ────── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(page_title="TrafficIQ", page_icon="🚦", layout="wide")

# ────── ATTRACTIVE CSS ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .block-container { padding: 1rem 1.5rem 0.5rem !important; max-width: 1300px; }
    .stButton button {
        background: linear-gradient(90deg, #f5a623, #ffcc6b);
        color: #0a0c10; font-weight: 700; border: none; border-radius: 10px;
        padding: 0.6rem; transition: 0.2s;
    }
    .stButton button:hover { transform: scale(1.01); opacity: 0.9; }
    .result-card {
        border-radius: 20px; padding: 1.2rem; text-align: center;
        backdrop-filter: blur(4px); margin-bottom: 1rem;
    }
    .metric-tile {
        background: #1e1e2a; border-radius: 16px; padding: 0.8rem;
        text-align: center; border: 1px solid #2d2d3a;
    }
    .zone-badge {
        background: #2d2d3a; border-radius: 20px; padding: 0.2rem 0.8rem;
        font-size: 0.7rem; color: #f5a623;
    }
    hr { margin: 0.5rem 0; background: #2d2d3a; }
    .prob-row { margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ────── LOAD MODEL & SCALER ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()
FEATURES = list(scaler.feature_names_in_)
CLASSES = list(model.classes_)

# ────── LOOKUP DATA ─────────────────────────────────────────
WEATHER_MAIN = ["Clear","Clouds","Drizzle","Fog","Haze","Mist","Rain","Smoke","Snow","Squall","Thunderstorm"]
WEATHER_DESC_MAP = {
    "Clear": ["sky is clear"],
    "Clouds": ["few clouds", "scattered clouds", "broken clouds", "overcast clouds"],
    "Rain": ["light rain", "moderate rain", "heavy intensity rain"],
    "Snow": ["snow", "light snow", "heavy snow"],
    "Thunderstorm": ["thunderstorm", "thunderstorm with rain"],
    "Drizzle": ["drizzle", "light intensity drizzle"],
    "Fog": ["fog"], "Haze": ["haze"], "Mist": ["mist"], "Smoke": ["smoke"], "Squall": ["squall"]
}
HOLIDAYS = ["None", "Christmas Day", "Columbus Day", "Independence Day", "Labor Day",
            "Martin Luther King Jr Day", "Memorial Day", "New Years Day", "State Fair",
            "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"]
ZONE_TRAITS = {
    "Urban Core": ("High density", "Signal timing", "Grid layout"),
    "Suburban": ("Medium density", "Arterial roads", "School zones"),
    "Highway": ("Variable speed", "Merge bottlenecks", "Ramp control"),
    "Industrial": ("Heavy vehicles", "Morning peaks", "Port activity"),
    "Airport Corridor": ("Flight schedules", "Drop-off surges", "Shuttle lanes")
}

# ────── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 TrafficIQ")
    st.caption("Smart congestion predictor")
    hour = st.slider("Hour (0–23)", 0, 23, 8)
    holiday = st.selectbox("Holiday", HOLIDAYS)
    temp_c = st.slider("Temp (°C)", -30, 45, 15)
    rain = st.number_input("Rain (mm)", 0.0, 100.0, 0.0, step=0.5)
    snow = st.number_input("Snow (mm)", 0.0, 20.0, 0.0, step=0.1)
    clouds = st.slider("Clouds (%)", 0, 100, 50)
    weather_main = st.selectbox("Weather", WEATHER_MAIN)
    desc_options = WEATHER_DESC_MAP.get(weather_main, ["sky is clear"])
    weather_desc = st.selectbox("Detail", desc_options)
    route_zone = st.selectbox("Zone", list(ZONE_TRAITS.keys()))
    route_km = st.number_input("Distance (km)", 1.0, 100.0, 12.0, step=0.5)
    predict = st.button("▶ RUN PREDICTION", use_container_width=True)

# ────── FEATURE BUILDER ────────────────────────────────────
def build_features(temp_c, rain, snow, clouds, hour, holiday, weather_main, weather_desc):
    temp = temp_c + 273.15
    data = np.zeros((1, len(FEATURES)))
    idx = {f: i for i, f in enumerate(FEATURES)}
    data[0, idx['temp']] = temp
    data[0, idx['rain_1h']] = rain
    data[0, idx['snow_1h']] = snow
    data[0, idx['clouds_all']] = clouds
    data[0, idx['hour']] = hour
    if holiday != "None":
        hkey = f"holiday_{holiday}"
        if hkey in idx: data[0, idx[hkey]] = 1
    wmkey = f"weather_main_{weather_main}"
    if wmkey in idx: data[0, idx[wmkey]] = 1
    wdkey = f"weather_description_{weather_desc}"
    if wdkey in idx: data[0, idx[wdkey]] = 1
    return data

# ────── MAIN CONTENT ───────────────────────────────────────
if predict:
    with st.spinner("Analyzing traffic patterns..."):
        time.sleep(0.2)
        X_raw = build_features(temp_c, rain, snow, clouds, hour, holiday, weather_main, weather_desc)
        X_scaled = scaler.transform(X_raw)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

    # Color mapping
    colors = {"High": "#ff4b4b", "Medium": "#f5a623", "Low": "#00e5a0"}
    color = colors[pred]
    icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[pred]

    # Result card
    st.markdown(f"""
    <div class="result-card" style="background: {color}10; border: 2px solid {color};">
        <div style="font-size: 3rem;">{icon}</div>
        <h1 style="margin:0; color:{color};">{pred.upper()} CONGESTION</h1>
        <p style="margin:0;">Gradient Boosting prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # Probabilities with bars
    st.markdown("#### 📊 Class probabilities")
    for cls, p in zip(CLASSES, proba):
        pct = p * 100
        st.markdown(f"""
        <div class="prob-row">
            <div style="display:flex; justify-content:space-between;">
                <span><b>{cls}</b></span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background:#2d2d3a; border-radius:8px;">
                <div style="width:{pct:.1f}%; background:{colors[cls]}; height:8px; border-radius:8px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Travel estimation
    base_speed = {"Low": 60, "Medium": 35, "High": 15}[pred]
    if rain > 5: base_speed *= 0.85
    if snow > 0.5: base_speed *= 0.70
    if 7 <= hour <= 9 or 16 <= hour <= 19: base_speed *= 0.80
    if holiday != "None": base_speed *= 0.90
    est_min = (route_km / max(base_speed, 5)) * 60
    delay = {"High":"15–40+ min", "Medium":"5–12 min", "Low":"0–2 min"}[pred]

    col1, col2, col3 = st.columns(3)
    col1.metric("⏱️ Estimated time", f"{est_min:.0f} min")
    col2.metric("🚗 Average speed", f"{base_speed:.0f} km/h")
    col3.metric("⚠️ Delay range", delay)

    # Zone insights
    traits = ZONE_TRAITS[route_zone]
    st.markdown(f"""
    <div class="metric-tile" style="margin-top:0.5rem;">
        <span class="zone-badge">🗺️ {route_zone}</span><br>
        <span style="font-size:0.85rem;">{traits[0]} · {traits[1]} · {traits[2]}</span>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    advice_text = {
        "High": "⚠️ Avoid peak routes. Delay travel 30–60 min.",
        "Medium": "⚡ Moderate congestion, allow extra buffer time.",
        "Low": "✅ Smooth flow, proceed as planned."
    }[pred]
    st.info(f"**📌 Advisory:** {advice_text}")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
        <div style="font-size: 4rem;">🚦</div>
        <h2>TrafficIQ — Real‑time congestion intelligence</h2>
        <p style="color: #aaa;">Adjust parameters in the sidebar and click <b>RUN PREDICTION</b></p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("🧠 About the model"):
        st.markdown("""
        - **Algorithm:** Gradient Boosting Classifier (GBM)
        - **Features:** 65 (weather, time, holiday, one‑hot encoding)
        - **Output:** High / Medium / Low congestion
        - **Plus:** Travel time regression, zone clustering
        """)

# Footer
st.markdown("""
<hr>
<div style="text-align: center; font-size: 0.7rem; color: #666;">
    TrafficIQ · ML-powered traffic prediction
</div>
""", unsafe_allow_html=True)
