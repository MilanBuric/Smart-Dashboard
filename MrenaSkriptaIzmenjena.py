import streamlit as st
import time
import pandas as pd
import requests
import psutil
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# ----------------- CONFIG ----------------- #
st.set_page_config(page_title='Smart Dashboard', layout="wide")

# ----------------- FUNCTIONS ----------------- #
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def calculate_dynamic_threshold(series, factor=2.0):
    return series.mean() + factor * series.std()

def fetch_weather(city_coords):
    results = []
    for city, coords in city_coords.items():
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['latitude']}&longitude={coords['longitude']}&hourly=temperature_2m"
            response = requests.get(url).json()
            temps = response.get("hourly", {}).get("temperature_2m", [])[:5]
            for temp in temps:
                results.append({"Location": city, "Temperature (°C)": round(temp)})
        except:
            results.append({"Location": city, "Temperature (°C)": "Error"})
    return pd.DataFrame(results)

def automatic_feature_optimization(df, target, k):
    num_df = df.select_dtypes(include=['number'])
    if target not in num_df or num_df.shape[0] < 2:
        return None, None
    X, y = num_df.drop(columns=target), num_df[target]
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit(X, y)
    scores = pd.DataFrame({
        "Feature": X.columns,
        "Score": selector.scores_
    }).sort_values(by="Score", ascending=False)
    return scores, scores["Feature"].head(k).tolist()

def update_system_metrics():
    return {
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent
    }

# ----------------- LOAD DATA ----------------- #
try:
    df = load_data("demo_data.csv")
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ----------------- UI ELEMENTS ----------------- #
cols = st.columns(6)
Voltage_chart = cols[0].line_chart()
Freq_chart = cols[1].line_chart()
Power_chart = cols[2].line_chart()
PA_chart = cols[3].line_chart()
CP_chart = cols[4].line_chart()
PF_chart = cols[5].line_chart()

# ----------------- WEATHER ----------------- #
st.markdown("### Weather Data")
weather_df = fetch_weather({
    "Zrenjanin, Serbia": {"latitude": 45.3755, "longitude": 20.4020},
    "Belgrade, Serbia": {"latitude": 44.8176, "longitude": 20.4633},
    "Novi Sad, Serbia": {"latitude": 45.2671, "longitude": 19.8335}
})
st.dataframe(weather_df)

# ----------------- FEATURE SELECTION ----------------- #
numeric_columns = df.select_dtypes(include='number').columns.tolist()
if numeric_columns:
    st.markdown("### Feature Optimization")
    target_column = st.selectbox("Target:", numeric_columns)
    k_val = st.slider("Select K features", 1, len(numeric_columns)-1, 3)
else:
    st.warning("No numeric data found.")
    st.stop()

afo_placeholder = st.empty()

# ----------------- MONITOR & LIVE LOOP ----------------- #
df_live = pd.DataFrame(columns=df.columns)
for i, row in df.iterrows():
    # Real-time data visualization updates
    if i % 5 == 0:
        Voltage_chart.add_rows({"Voltage": [row.get("Voltage", None)], "Current": [row.get("Current", None)]})
        Freq_chart.add_rows({"Measured_Frequency": [row.get("Measured_Frequency", None)]})
        Power_chart.add_rows({
            "Active_Power": [row.get("Active_Power", None)],
            "Reactive_Power": [row.get("Reactive_Power", None)],
            "Apperent_Power": [row.get("Apperent_Power", None)]
        })
        PA_chart.add_rows({"Phase_Voltage_Angle": [row.get("Phase_Voltage_Angle", None)]})
        CP_chart.add_rows({"Cos_Phi": [row.get("Cos_Phi", None)]})
        PF_chart.add_rows({"Power_Factor": [row.get("Power_Factor", None)]})
    
    # Real-time system performance
    if i % 10 == 0:
        metrics = update_system_metrics()
        st.sidebar.metric("CPU Usage", f"{metrics['cpu']}%")
        st.sidebar.metric("Memory Usage", f"{metrics['memory']}%")
        st.sidebar.metric("Disk Usage", f"{metrics['disk']}%")

    # Real-time feature optimization
    df_live = pd.concat([df_live, pd.DataFrame([row])], ignore_index=True)
    if len(df_live) % 10 == 0:
        result_df, selected_features = automatic_feature_optimization(df_live, target_column, k_val)
        if result_df is not None:
            with afo_placeholder.container():
                st.markdown("#### Selected Features")
                st.dataframe(result_df)

    time.sleep(0.1)  # optional to simulate real-time streaming
