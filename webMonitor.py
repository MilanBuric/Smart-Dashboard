import streamlit as st
import time
import pandas as pd
import requests
import psutil
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Set Streamlit page configuration
st.set_page_config(
    page_title='Smart Dashboard',
    layout="wide",
)

# ----------------- DYNAMIC THRESHOLD FUNCTION ----------------- #
def calculate_dynamic_threshold(series, factor=2.0):
    """
    Calculate a dynamic threshold for a given pandas Series.
    Default uses mean + (factor * std).
    """
    return series.mean() + factor * series.std()

# ----------------- INITIAL UI SETUP ----------------- #
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

# Define subheaders for real-time charts
col1.subheader("Voltage [V] and Current")
Voltage = col1.line_chart({"Voltage": [], "Current": []})

col2.subheader("Frequency")
Freq = col2.line_chart({"Measured_Frequency": []})

col3.subheader("Power Data")
AP = col3.line_chart({
    "Active_Power": [], "Reactive_Power": [], "Apperent_Power": []
})

col4.subheader("Angle Data")
PA = col4.line_chart({
    "Phase_Voltage_Angle": []
})

col5.subheader("Cos Phi Data")
CP = col5.line_chart({"Cos_Phi": []})

col6.subheader("Power Factor Data")
PF = col6.line_chart({"Power_Factor": []})

# Placeholders for Date and Time metrics
YMDt = st.empty()
Tt = st.empty()

# ----------------- LOAD DATASET ----------------- #
try:
    df = pd.read_csv("demo_data.csv")
except FileNotFoundError:
    st.error("Data file 'demo_data.csv' not found. Please ensure it is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# Display dataset columns
col_dataset, col_updated = st.columns(2)
col_dataset.markdown("### Columns in dataset:")
col_dataset.table(pd.DataFrame(df.columns.tolist(), columns=["Columns"]))

initial_message = col_updated.info("The list of updated columns with their last known values will be displayed here after processing is complete.")

# ----------------- FILTER COLUMNS SECTION ----------------- #
st.markdown("### Filter Columns")
selected_columns = st.multiselect("Select columns to display:", options=df.columns)
if selected_columns:
    st.write("Filtered Columns Data:")
    st.table(df[selected_columns])

# ----------------- DYNAMICALLY MOVING VOLTAGE SLIDER ----------------- #
st.markdown("### Dynamic Voltage Threshold (Real-Time)")
slider_placeholder = st.empty()  # We'll use this placeholder to update the slider each row

# Pre-calculate a threshold if needed (optional, for reference)
if "Voltage" in df.columns:
    dynamic_voltage_calc = calculate_dynamic_threshold(df["Voltage"], factor=2.0)
    st.write(f"Pre-calculated threshold (mean + 2*std): {dynamic_voltage_calc:.2f} V")
else:
    st.error("No 'Voltage' column found in dataset. Slider won't update.")

# ----------------- WEATHER API INTEGRATION ----------------- #
st.markdown("### Weather Data")
cities = {
    "Zrenjanin, Serbia": {"latitude": 45.3755, "longitude": 20.4020},
    "Belgrade, Serbia": {"latitude": 44.8176, "longitude": 20.4633},
    "Novi Sad, Serbia": {"latitude": 45.2671, "longitude": 19.8335},
    "Banja Luka, Bosnia and Herzegovina": {"latitude": 44.7722, "longitude": 17.1910},
    "Sarajevo, Bosnia and Herzegovina": {"latitude": 43.8486, "longitude": 18.3564},
    "Zagreb, Croatia": {"latitude": 45.8125, "longitude": 15.978}
}

try:
    weather_data_list = []
    for city, coords in cities.items():
        latitude = coords["latitude"]
        longitude = coords["longitude"]
        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m"
        ).json()

        if "hourly" in weather and "temperature_2m" in weather["hourly"]:
            temperatures = weather["hourly"]["temperature_2m"][:5]
            if temperatures:
                for temp in temperatures:
                    weather_data_list.append({
                        "Location": city,
                        "Temperature (°C)": round(temp)
                    })
        else:
            st.error(f"Error fetching data for {city}.")
    
    if weather_data_list:
        weather_data = pd.DataFrame(weather_data_list)
        st.table(weather_data)
    else:
        st.error("No weather data found.")
        
except Exception as e:
    st.error(f"Unable to fetch weather data: {e}")

# ----------------- REAL-TIME PERFORMANCE MONITOR ----------------- #
st.markdown("### Performance Monitor")
perf_col1, perf_col2, perf_col3 = st.columns(3)
perf_col1.subheader("CPU Usage")
cpu_chart = perf_col1.line_chart({"CPU Usage": []})

perf_col2.subheader("Memory Usage")
memory_chart = perf_col2.line_chart({"Memory Usage": []})

perf_col3.subheader("Disk Usage")
disk_chart = perf_col3.line_chart({"Disk Usage": []})

with st.expander("System performance overview"):
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    disk_usage_info = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()

    st.write(f"**CPU usage:** {cpu_usage}%")
    st.write(f"**RAM usage:** {memory_info.percent}%")
    st.write(f"**Total RAM memory:** {memory_info.total / (1024 ** 3):.2f} GB")
    st.write(f"**Available RAM memory:** {memory_info.available / (1024 ** 3):.2f} GB")
    st.write(f"**Disk usage:** {disk_usage_info.percent}%")
    st.write(f"**Total space on the Disk:** {disk_usage_info.total / (1024 ** 3):.2f} GB")
    st.write(f"**Available memory on Disk:** {disk_usage_info.free / (1024 ** 3):.2f} GB")
    st.write(f"**Sent data:** {net_io.bytes_sent / (1024 ** 2):.2f} MB")
    st.write(f"**Data received:** {net_io.bytes_recv / (1024 ** 2):.2f} MB")

def update_performance_metrics():
    cpu_chart.add_rows({"CPU Usage": [psutil.cpu_percent()]})
    memory_chart.add_rows({"Memory Usage": [psutil.virtual_memory().percent]})
    disk_chart.add_rows({"Disk Usage": [psutil.disk_usage('/').percent]})

def update_real_time_charts(row):
    updated_columns = {}

    if "Voltage" in row and "Current" in row:
        Voltage.add_rows([{"Voltage": row["Voltage"], "Current": row["Current"]}])
        updated_columns["Voltage"] = row["Voltage"]
        updated_columns["Current"] = row["Current"]

    if "Measured_Frequency" in row:
        Freq.add_rows([{"Measured_Frequency": row["Measured_Frequency"]}])
        updated_columns["Measured_Frequency"] = row["Measured_Frequency"]

    if all(k in row for k in ["Active_Power", "Reactive_Power", "Apperent_Power"]):
        AP.add_rows([{
            "Active_Power": row["Active_Power"],
            "Reactive_Power": row["Reactive_Power"],
            "Apperent_Power": row["Apperent_Power"]
        }])
        updated_columns["Active_Power"] = row["Active_Power"]
        updated_columns["Reactive_Power"] = row["Reactive_Power"]
        updated_columns["Apperent_Power"] = row["Apperent_Power"]

    if "Phase_Voltage_Angle" in row:
        PA.add_rows([{"Phase_Voltage_Angle": row["Phase_Voltage_Angle"]}])
        updated_columns["Phase_Voltage_Angle"] = row["Phase_Voltage_Angle"]

    if "Cos_Phi" in row:
        CP.add_rows([{"Cos_Phi": row["Cos_Phi"]}])
        updated_columns["Cos_Phi"] = row["Cos_Phi"]

    if "Power_Factor" in row:
        PF.add_rows([{"Power_Factor": row["Power_Factor"]}])
        updated_columns["Power_Factor"] = row["Power_Factor"]
    
    return updated_columns

# ----------------- REAL-TIME LOOP ----------------- #
all_updated_columns = {}
for i, row in df.iterrows():
    # Update the slider for each incoming voltage value (if present)
    if "Voltage" in row:
        with slider_placeholder.container():
            st.slider(
                "Voltage Threshold (Live)",
                min_value=0.0,
                max_value=300.0,
                step=0.1,
                value=float(row["Voltage"]),
                key=f"voltage_slider_{i}"  # unique key each iteration
            )

    # Update performance metrics
    update_performance_metrics()

    # Update charts with current row
    updated_columns = update_real_time_charts(row)
    all_updated_columns.update(updated_columns)

    # Update Date and Time metrics
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
        relevant_columns = [
            "Voltage", "Current", "Measured_Frequency", "Active_Power",
            "Reactive_Power", "Apperent_Power", "Phase_Voltage_Angle",
            "Cos_Phi", "Power_Factor"
        ]
        formatted_updated_columns = [
            {"Column": col, "Last Known Value": value}
            for col, value in all_updated_columns.items()
            if col in relevant_columns
        ]
        updated_columns_df = pd.DataFrame(formatted_updated_columns)
        col_updated.table(updated_columns_df)
except Exception as e:
    st.error(f"An error occurred while displaying updated columns: {e}")

# ----------------- HISTORIC DATA ANALYSIS ----------------- #
st.markdown("### Historic Data Analysis")
df['Time'] = pd.to_datetime(df['Time'], errors='coerce').dt.time
start_time = st.time_input("Start Time", value=df["Time"].min())
end_time = st.time_input("End Time", value=df["Time"].max())
df_filtered = df[(df["Time"] >= start_time) & (df["Time"] <= end_time)]

# Display historical data charts
chart_col1, chart_col2 = st.columns(2)
chart_col1.subheader("Voltage [V] and Current")
chart_col1.line_chart(df_filtered[["Voltage", "Current"]])

chart_col2.subheader("Power Data (Active, Reactive, Apparent)")
chart_col2.line_chart(df_filtered[["Active_Power", "Reactive_Power", "Apperent_Power"]])

st.markdown("### Summary of Selected Data Range")
st.write(df_filtered.describe())

# Display filtered data
st.write("Filtered Data:")
st.dataframe(df_filtered)

# ----------------- AUTOMATIC FEATURE OPTIMIZATION ----------------- #
st.markdown("### Automatic Feature Optimization")
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_columns:
    target_column = st.selectbox("Select target column for optimization:", options=numeric_columns, index=0)
    k = st.slider("Number of features to select", min_value=1, max_value=len(numeric_columns)-1, value=min(3, len(numeric_columns)-1))
    
    def automatic_feature_optimization(dataframe, target, k):
        dataframe_numeric = dataframe.select_dtypes(include=['int64', 'float64'])
        if target not in dataframe_numeric.columns:
            st.error("Target column must be numeric for optimization.")
            return None, None
        X = dataframe_numeric.drop(columns=[target])
        y = dataframe_numeric[target]
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        scores = selector.scores_
        feature_names = X.columns
        result_df = pd.DataFrame({
            "Feature": feature_names,
            "Score": scores
        }).sort_values(by="Score", ascending=False)
        selected_features = result_df.head(k)["Feature"].tolist()
        return result_df, selected_features
    
    result_df, selected_features = automatic_feature_optimization(df, target_column, k)
    if result_df is not None:
        st.write("Feature scores for all numeric columns:")
        st.dataframe(result_df)
        st.write("Selected Features:")
        st.write(selected_features)
else:
    st.warning("No numeric columns available for optimization.")
