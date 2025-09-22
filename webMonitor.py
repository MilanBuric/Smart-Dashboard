import streamlit as st
import time
import pandas as pd
import requests
import psutil
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# ----------------- CACHE DATA LOADING ----------------- #
@st.cache_data
def load_dataset(path="demo_data.csv"):
    return pd.read_csv(path)

@st.cache_data
def fetch_weather_data(cities):
    weather_data_list = []
    for city, coords in cities.items():
        latitude = coords["latitude"]
        longitude = coords["longitude"]
        try:
            resp = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m"
            )
            data = resp.json()
            if "hourly" in data and "temperature_2m" in data["hourly"]:
                temperatures = data["hourly"]["temperature_2m"][:5]
                for temp in temperatures:
                    weather_data_list.append({
                        "Location": city,
                        "Temperature (°C)": round(temp)
                    })
        except Exception as e:
            st.error(f"Error fetching data for {city}: {e}")
    if weather_data_list:
        return pd.DataFrame(weather_data_list)
    else:
        return None

# ----------------- DYNAMIC THRESHOLD FUNCTION ----------------- #
def calculate_dynamic_threshold(series, factor=2.0):
    return series.mean() + factor * series.std()

# ----------------- UI INITIALIZATION ----------------- #
st.set_page_config(page_title='Smart Dashboard', layout="wide")

# Layout columns for charts
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

# Real-time chart placeholders
Voltage = col1.line_chart({"Voltage": [], "Current": []})
Freq = col2.line_chart({"Measured_Frequency": []})
AP = col3.line_chart({"Active_Power": [], "Reactive_Power": [], "Apperent_Power": []})
PA = col4.line_chart({"Phase_Voltage_Angle": []})
CP = col5.line_chart({"Cos_Phi": []})
PF = col6.line_chart({"Power_Factor": []})

# Placeholders for Date and Time metrics
YMDt = st.empty()
Tt = st.empty()

# ----------------- LOAD AND DISPLAY DATASET ----------------- #
try:
    df = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

col_dataset, col_updated = st.columns(2)
col_dataset.markdown("### Columns in dataset:")
col_dataset.table(pd.DataFrame(df.columns.tolist(), columns=["Columns"]))
initial_message = col_updated.info("The list of updated columns with their last known values will be displayed here after processing is complete.")

# ----------------- FILTER COLUMNS ----------------- #
st.markdown("### Filter Columns")
selected_columns = st.multiselect("Select columns to display:", options=df.columns)
if selected_columns:
    st.write("Filtered Columns Data:")
    st.table(df[selected_columns])

# ----------------- DYNAMIC VOLTAGE SLIDER ----------------- #
st.markdown("### Dynamic Voltage Threshold (Real-Time)")
slider_placeholder = st.empty()
if "Voltage" in df.columns:
    dynamic_voltage_calc = calculate_dynamic_threshold(df["Voltage"], factor=2.0)
    st.write(f"Pre-calculated threshold (mean + 2*std): {dynamic_voltage_calc:.2f} V")
else:
    st.error("No 'Voltage' column found in dataset.")

# ----------------- WEATHER DATA ----------------- #
st.markdown("### Weather Data")
cities = {
    "Zrenjanin, Serbia": {"latitude": 45.3755, "longitude": 20.4020},
    "Belgrade, Serbia": {"latitude": 44.8176, "longitude": 20.4633},
    "Novi Sad, Serbia": {"latitude": 45.2671, "longitude": 19.8335},
    "Banja Luka, Bosnia and Herzegovina": {"latitude": 44.7722, "longitude": 17.1910},
    "Sarajevo, Bosnia and Herzegovina": {"latitude": 43.8486, "longitude": 18.3564},
    "Zagreb, Croatia": {"latitude": 45.8125, "longitude": 15.978}
}
weather_df = fetch_weather_data(cities)
if weather_df is not None:
    st.table(weather_df)
else:
    st.error("No weather data found.")

# ----------------- PERFORMANCE MONITOR ----------------- #
st.markdown("### Performance Monitor")
perf_col1, perf_col2, perf_col3 = st.columns(3)
cpu_chart = perf_col1.line_chart({"CPU Usage": []})
memory_chart = perf_col2.line_chart({"Memory Usage": []})
disk_chart = perf_col3.line_chart({"Disk Usage": []})

with st.expander("System performance overview"):
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()
    st.write(f"**CPU usage:** {cpu_usage}%")
    st.write(f"**RAM usage:** {memory_info.percent}%")
    st.write(f"**Total RAM:** {memory_info.total / (1024**3):.2f} GB")
    st.write(f"**Available RAM:** {memory_info.available / (1024**3):.2f} GB")
    st.write(f"**Disk usage:** {disk_info.percent}%")
    st.write(f"**Total Disk Space:** {disk_info.total / (1024**3):.2f} GB")
    st.write(f"**Available Disk Space:** {disk_info.free / (1024**3):.2f} GB")
    st.write(f"**Sent data:** {net_io.bytes_sent / (1024**2):.2f} MB")
    st.write(f"**Received data:** {net_io.bytes_recv / (1024**2):.2f} MB")

def update_performance_metrics():
    cpu_chart.add_rows({"CPU Usage": [psutil.cpu_percent()]})
    memory_chart.add_rows({"Memory Usage": [psutil.virtual_memory().percent]})
    disk_chart.add_rows({"Disk Usage": [psutil.disk_usage('/').percent]})

def update_real_time_charts(row):
    updated = {}
    if "Voltage" in row and "Current" in row:
        Voltage.add_rows([{"Voltage": row["Voltage"], "Current": row["Current"]}])
        updated["Voltage"] = row["Voltage"]
        updated["Current"] = row["Current"]
    if "Measured_Frequency" in row:
        Freq.add_rows([{"Measured_Frequency": row["Measured_Frequency"]}])
        updated["Measured_Frequency"] = row["Measured_Frequency"]
    if all(k in row for k in ["Active_Power", "Reactive_Power", "Apperent_Power"]):
        AP.add_rows([{
            "Active_Power": row["Active_Power"],
            "Reactive_Power": row["Reactive_Power"],
            "Apperent_Power": row["Apperent_Power"]
        }])
        updated["Active_Power"] = row["Active_Power"]
        updated["Reactive_Power"] = row["Reactive_Power"]
        updated["Apperent_Power"] = row["Apperent_Power"]
    if "Phase_Voltage_Angle" in row:
        PA.add_rows([{"Phase_Voltage_Angle": row["Phase_Voltage_Angle"]}])
        updated["Phase_Voltage_Angle"] = row["Phase_Voltage_Angle"]
    if "Cos_Phi" in row:
        CP.add_rows([{"Cos_Phi": row["Cos_Phi"]}])
        updated["Cos_Phi"] = row["Cos_Phi"]
    if "Power_Factor" in row:
        PF.add_rows([{"Power_Factor": row["Power_Factor"]}])
        updated["Power_Factor"] = row["Power_Factor"]
    return updated

# ----------------- LIVE AUTOMATIC FEATURE OPTIMIZATION SETUP ----------------- #
st.markdown("### Automatic Feature Optimization (Live)")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
afo_placeholder = st.empty()
target_column = st.selectbox("Select target column for optimization:", options=numeric_cols, index=0)
k = st.slider("Number of features to select", min_value=1, max_value=len(numeric_cols)-1, value=min(3, len(numeric_cols)-1))
df_live = pd.DataFrame(columns=df.columns)

def automatic_feature_optimization(dataframe, target, k):
    num_df = dataframe.select_dtypes(include=['int64', 'float64'])
    if target not in num_df.columns or num_df.empty:
        return None, None
    X = num_df.drop(columns=[target])
    y = num_df[target]
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit(X, y)
    scores = selector.scores_
    result_df = pd.DataFrame({"Feature": X.columns, "Score": scores}).sort_values(by="Score", ascending=False)
    selected_features = result_df.head(k)["Feature"].tolist()
    return result_df, selected_features

# ----------------- REAL-TIME LOOP ----------------- #
all_updated_columns = {}
for i, row in df.iterrows():
    # Update live voltage slider
    if "Voltage" in row:
        with slider_placeholder.container():
            st.slider(
                "Voltage Threshold (Live)",
                min_value=0.0,
                max_value=300.0,
                step=0.1,
                value=float(row["Voltage"]),
                key=f"voltage_slider_{i}"
            )
    update_performance_metrics()
    updated = update_real_time_charts(row)
    all_updated_columns.update(updated)
    
    # Update live dataset for AFO
    df_live = pd.concat([df_live, pd.DataFrame([row])], ignore_index=True)
    if len(df_live) % 10 == 0:
        result_df, selected_features = automatic_feature_optimization(df_live, target_column, k)
        if result_df is not None:
            with afo_placeholder.container():
                st.markdown("#### Live Automatic Feature Optimization Results")
                st.dataframe(result_df)
                st.write("Selected Features:", selected_features)
    
    if "Date" in row:
        YMDt.metric("Date", row["Date"])
    if "Time" in row:
        Tt.metric("Time", row["Time"])
    time.sleep(0.1)

# ----------------- DISPLAY UPDATED COLUMNS ----------------- #
try:
    initial_message.empty()
    with st.container():
        col_updated.markdown("### Updated Columns with Last Known Values:")
        relevant = ["Voltage", "Current", "Measured_Frequency", "Active_Power",
                    "Reactive_Power", "Apperent_Power", "Phase_Voltage_Angle",
                    "Cos_Phi", "Power_Factor"]
        formatted = [{"Column": col, "Last Known Value": value}
                     for col, value in all_updated_columns.items() if col in relevant]
        col_updated.table(pd.DataFrame(formatted))
except Exception as e:
    st.error(f"Error displaying updated columns: {e}")

# ----------------- HISTORIC DATA ANALYSIS ----------------- #
st.markdown("### Historic Data Analysis")
df['Time'] = pd.to_datetime(df['Time'], errors='coerce').dt.time
start_time = st.time_input("Start Time", value=df["Time"].min())
end_time = st.time_input("End Time", value=df["Time"].max())
df_filtered = df[(df["Time"] >= start_time) & (df["Time"] <= end_time)]

chart_col1, chart_col2 = st.columns(2)
chart_col1.subheader("Voltage [V] and Current")
chart_col1.line_chart(df_filtered[["Voltage", "Current"]])
chart_col2.subheader("Power Data (Active, Reactive, Apparent)")
chart_col2.line_chart(df_filtered[["Active_Power", "Reactive_Power", "Apperent_Power"]])

st.markdown("### Summary of Selected Data Range")
st.write(df_filtered.describe())
st.write("Filtered Data:")
st.dataframe(df_filtered)