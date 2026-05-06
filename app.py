
import streamlit as st
import pickle
import numpy as np
import time

# ────── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="TrafficIQ - GBM Congestion Predictor",
    page_icon="🚦",
    layout="wide"
)

# ────── MODERN CSS ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .block-container { padding: 1.5rem 2rem 1rem !important; max-width: 1350px; }
    
    .stButton button {
        background: linear-gradient(90deg, #f5a623, #ffcc6b);
        color: #0a0c10;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(245, 166, 35, 0.4);
    }
    
    .result-card {
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    
    .metric-tile {
        background: #1e1e2a;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2d2d3a;
        height: 100%;
    }
    
    .zone-badge {
        background: #2d2d3a;
        border-radius: 30px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        color: #f5a623;
        font-weight: 500;
    }
    
    hr { margin: 1.2rem 0; background: #2d2d3a; }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 1rem;
    }
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
    st.caption("**Traffic Congestion Prediction using GBM Ensemble Learning**")
    
    st.markdown("### 📌 Input Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour of Day", 0, 23, 8)
    with col2:
        holiday = st.selectbox("Holiday", HOLIDAYS)
    
    st.markdown("### 🌤️ Weather Conditions")
    temp_c = st.slider("Temperature (°C)", -30, 45, 15)
    rain = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
    snow = st.number_input("Snowfall (mm)", 0.0, 20.0, 0.0, step=0.1)
    clouds = st.slider("Cloud Cover (%)", 0, 100, 50)
    
    weather_main = st.selectbox("Weather Category", WEATHER_MAIN)
    desc_options = WEATHER_DESC_MAP.get(weather_main, ["sky is clear"])
    weather_desc = st.selectbox("Weather Description", desc_options)
    
    st.markdown("### 🛣️ Route Details")
    route_zone = st.selectbox("Traffic Zone", list(ZONE_TRAITS.keys()))
    route_km = st.number_input("Distance (km)", 1.0, 100.0, 12.0, step=0.5)
    
    st.markdown("---")
    predict = st.button("🔍 PREDICT CONGESTION", use_container_width=True, type="primary")

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
    with st.spinner("Running GBM Ensemble Model..."):
        time.sleep(0.35)
        X_raw = build_features(temp_c, rain, snow, clouds, hour, holiday, weather_main, weather_desc)
        X_scaled = scaler.transform(X_raw)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

    # Color mapping
    colors = {"High": "#ff4b4b", "Medium": "#f5a623", "Low": "#00e5a0"}
    color = colors[pred]
    icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[pred]

    # Main Result Card
    st.markdown(f"""
    <div class="result-card" style="background: {color}15; border: 2px solid {color};">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">{icon}</div>
        <h1 style="margin:0; color:{color}; font-size: 2.6rem; font-weight: 700;">{pred.upper()} TRAFFIC CONGESTION</h1>
        <p style="margin:0.5rem 0 0; font-size: 1.1rem; opacity: 0.95;">
            Gradient Boosting Machine (GBM) Ensemble Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Prediction Confidence
    st.markdown('<p class="section-title">📊 PREDICTION CONFIDENCE (GBM Probabilities)</p>', unsafe_allow_html=True)
    for cls, p in zip(CLASSES, proba):
        pct = p * 100
        st.markdown(f"""
        <div style="margin-bottom: 0.9rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 4px;">
                <span><b>{cls} Traffic</b></span>
                <span><b>{pct:.1f}%</b></span>
            </div>
            <div style="background:#2d2d3a; border-radius:10px; height:11px;">
                <div style="width:{pct:.1f}%; background:{colors[cls]}; height:11px; border-radius:10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Travel Analytics
    st.markdown('<p class="section-title">⏱️ TRAVEL ANALYTICS</p>', unsafe_allow_html=True)
    
    base_speed = {"Low": 60, "Medium": 35, "High": 15}[pred]
    if rain > 5: base_speed *= 0.85
    if snow > 0.5: base_speed *= 0.70
    if 7 <= hour <= 9 or 16 <= hour <= 19: base_speed *= 0.80
    if holiday != "None": base_speed *= 0.90
    
    est_min = (route_km / max(base_speed, 5)) * 60
    delay = {"High":"15–40+ min", "Medium":"5–12 min", "Low":"0–2 min"}[pred]

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Travel Time", f"{est_min:.0f} minutes")
    col2.metric("Expected Average Speed", f"{base_speed:.0f} km/h")
    col3.metric("Expected Delay", delay)

    # Route Intelligence
    traits = ZONE_TRAITS[route_zone]
    st.markdown('<p class="section-title">🛣️ ROUTE & ZONE INTELLIGENCE</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-tile">
        <span class="zone-badge">📍 {route_zone}</span><br><br>
        <b>{traits[0]}</b> • {traits[1]} • {traits[2]}
    </div>
    """, unsafe_allow_html=True)

    # Traffic Advisory
    advice = {
        "High": "Heavy congestion predicted. Consider alternate route or delay travel.",
        "Medium": "Moderate traffic expected. Allow extra time for the journey.",
        "Low": "Smooth traffic conditions. Safe to travel as planned."
    }[pred]
    
    st.info(f"**🚦 Traffic Advisory:** {advice}")

else:
    # Welcome / Project Info
    st.markdown("""
    <div style="text-align: center; padding: 4rem 1rem 3rem;">
        <div style="font-size: 5.5rem; margin-bottom: 1.2rem;">🚦</div>
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Traffic Congestion Prediction</h1>
        <h2 style="color: #f5a623; font-weight: 500;">Using Gradient Boosting Machine (GBM) Ensemble Learning</h2>
        <p style="color: #aaaaaa; max-width: 700px; margin: 1.8rem auto; font-size: 1.1rem;">
            A B.Tech Project • Real-time prediction of traffic conditions based on weather, time and route data
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**Ensemble Learning**\nGradient Boosting Machine (GBM)")
    with col_b:
        st.info("**Multi-class Classification**\nLow / Medium / High Congestion")
    with col_c:
        st.info("**Feature Inputs**\nWeather + Temporal + Route Zone")

    with st.expander("🧠 About the Model", expanded=True):
        st.markdown("""
        - **Algorithm**: Gradient Boosting Machine (GBM)  
        - **Technique**: Ensemble Learning (Boosting)  
        - **Task**: Multi-class Classification (Low / Medium / High Traffic)  
        - **Features**: Weather conditions, Hour, Holiday, Route Zone  
        - **Output**: Congestion Level + Confidence Scores + Travel Estimation
        """)

# Footer
st.markdown("""
<hr>
<div style="text-align: center; font-size: 0.8rem; color: #666;">
    Traffic Congestion Prediction using GBM Ensemble Learning • B.Tech Semester Project
</div>
""", unsafe_allow_html=True)
