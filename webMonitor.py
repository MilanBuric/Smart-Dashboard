import streamlit as st
import time
import pandas as pd
import requests
import psutil
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# ----------------- CONFIGURATION ----------------- #
DATA_FILE = "demo_data.csv"
CITIES = {
    "Zrenjanin, Serbia": {"latitude": 45.3755, "longitude": 20.4020},
    "Belgrade, Serbia": {"latitude": 44.8176, "longitude": 20.4633},
    "Novi Sad, Serbia": {"latitude": 45.2671, "longitude": 19.8335},
    "Banja Luka, Bosnia and Herzegovina": {"latitude": 44.7722, "longitude": 17.1910},
    "Sarajevo, Bosnia and Herzegovina": {"latitude": 43.8486, "longitude": 18.3564},
    "Zagreb, Croatia": {"latitude": 45.8125, "longitude": 15.978}
}
RELEVANT_COLUMNS = [
    "Voltage", "Current", "Measured_Frequency", "Active_Power",
    "Reactive_Power", "Apperent_Power", "Phase_Voltage_Angle",
    "Cos_Phi", "Power_Factor"
]

# ----------------- HELPER FUNCTIONS ----------------- #

def calculate_dynamic_threshold(series, factor=2.0):
    """Calculate a dynamic threshold for a given pandas Series."""
    return series.mean() + factor * series.std()

@st.cache_data
def load_data(file_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce').dt.time
        return df
    except FileNotFoundError:
        st.error(f"Data file '{file_path}' not found. Please ensure it is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

def initialize_ui():
    """Initialize the Streamlit UI components."""
    st.set_page_config(page_title='Smart Dashboard', layout="wide")
    st.title("Smart Dashboard")

    # Placeholders for metrics
    metrics_placeholders = {
        "date": st.empty(),
        "time": st.empty()
    }

    # Chart columns
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    charts = {
        "Voltage": col1.line_chart({"Voltage": [], "Current": []}),
        "Freq": col2.line_chart({"Measured_Frequency": []}),
        "AP": col3.line_chart({"Active_Power": [], "Reactive_Power": [], "Apperent_Power": []}),
        "PA": col4.line_chart({"Phase_Voltage_Angle": []}),
        "CP": col5.line_chart({"Cos_Phi": []}),
        "PF": col6.line_chart({"Power_Factor": []})
    }
    
    col1.subheader("Voltage [V] and Current")
    col2.subheader("Frequency")
    col3.subheader("Power Data")
    col4.subheader("Angle Data")
    col5.subheader("Cos Phi Data")
    col6.subheader("Power Factor Data")

    # Performance monitor
    st.markdown("### Performance Monitor")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    perf_col1.subheader("CPU Usage")
    perf_col2.subheader("Memory Usage")
    perf_col3.subheader("Disk Usage")
    
    perf_charts = {
        "cpu": perf_col1.line_chart({"CPU Usage": []}),
        "memory": perf_col2.line_chart({"Memory Usage": []}),
        "disk": perf_col3.line_chart({"Disk Usage": []})
    }

    return charts, perf_charts, metrics_placeholders

def display_static_info(df):
    """Display static information like dataset columns and filters."""
    col_dataset, col_updated = st.columns(2)
    col_dataset.markdown("### Columns in dataset:")
    col_dataset.table(pd.DataFrame(df.columns.tolist(), columns=["Columns"]))
    
    initial_message = col_updated.info("The list of updated columns with their last known values will be displayed here after processing is complete.")

    st.markdown("### Filter Columns")
    selected_columns = st.multiselect("Select columns to display:", options=df.columns)
    if selected_columns:
        st.write("Filtered Columns Data:")
        st.table(df[selected_columns])
    
    return col_updated, initial_message

def display_voltage_threshold_info(df):
    """Display the dynamic voltage threshold slider and info."""
    st.markdown("### Dynamic Voltage Threshold (Real-Time)")
    slider_placeholder = st.empty()
    if "Voltage" in df.columns:
        dynamic_voltage_calc = calculate_dynamic_threshold(df["Voltage"], factor=2.0)
        st.write(f"Pre-calculated threshold (mean + 2*std): {dynamic_voltage_calc:.2f} V")
    else:
        st.error("No 'Voltage' column found in dataset. Slider won't update.")
    return slider_placeholder

def fetch_weather_data(cities):
    """Fetch and display weather data from Open-Meteo API."""
    st.markdown("### Weather Data")
    try:
        weather_data_list = []
        for city, coords in cities.items():
            url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['latitude']}&longitude={coords['longitude']}&hourly=temperature_2m"
            response = requests.get(url)
            response.raise_for_status()
            weather = response.json()

            if "hourly" in weather and "temperature_2m" in weather["hourly"]:
                temperatures = weather["hourly"]["temperature_2m"][:5]
                for temp in temperatures:
                    weather_data_list.append({"Location": city, "Temperature (°C)": round(temp)})
            else:
                st.warning(f"No weather data found for {city}.")
        
        if weather_data_list:
            st.table(pd.DataFrame(weather_data_list))
        else:
            st.error("Could not retrieve weather data for any city.")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to fetch weather data: {e}")
    except Exception as e:
        st.error(f"An error occurred while processing weather data: {e}")

def display_system_performance():
    """Display a detailed system performance overview in an expander."""
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

def update_performance_charts(perf_charts):
    """Update the real-time performance charts."""
    perf_charts["cpu"].add_rows({"CPU Usage": [psutil.cpu_percent()]})
    perf_charts["memory"].add_rows({"Memory Usage": [psutil.virtual_memory().percent]})
    perf_charts["disk"].add_rows({"Disk Usage": [psutil.disk_usage('/').percent]})

def update_real_time_charts(charts, row):
    """Update the real-time data charts from a data row."""
    updated_columns = {}
    
    def add_to_chart(chart_key, data_map):
        if all(k in row for k in data_map.keys()):
            charts[chart_key].add_rows([data_map])
            updated_columns.update(data_map)

    add_to_chart("Voltage", {"Voltage": row.get("Voltage"), "Current": row.get("Current")})
    add_to_chart("Freq", {"Measured_Frequency": row.get("Measured_Frequency")})
    add_to_chart("AP", {"Active_Power": row.get("Active_Power"), "Reactive_Power": row.get("Reactive_Power"), "Apperent_Power": row.get("Apperent_Power")})
    add_to_chart("PA", {"Phase_Voltage_Angle": row.get("Phase_Voltage_Angle")})
    add_to_chart("CP", {"Cos_Phi": row.get("Cos_Phi")})
    add_to_chart("PF", {"Power_Factor": row.get("Power_Factor")})
    
    return updated_columns

def automatic_feature_optimization(dataframe, target, k):
    """Perform automatic feature optimization using SelectKBest."""
    dataframe_numeric = dataframe.select_dtypes(include=['number'])
    if target not in dataframe_numeric.columns or dataframe_numeric.shape[0] < 2:
        return None, None
    
    X = dataframe_numeric.drop(columns=[target]).dropna(axis=1, how='all')
    y = dataframe_numeric[target]
    
    # Align X and y
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]

    if X.empty or y.empty:
        return None, None

    k = min(k, X.shape[1])
    if k == 0:
        return None, None
        
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit(X, y)
    
    result_df = pd.DataFrame({
        "Feature": X.columns,
        "Score": selector.scores_
    }).sort_values(by="Score", ascending=False)
    
    selected_features = result_df.head(k)["Feature"].tolist()
    return result_df, selected_features

def display_historical_data(df):
    """Display controls and charts for historical data analysis."""
    st.markdown("### Historic Data Analysis")
    
    min_time = df["Time"].min()
    max_time = df["Time"].max()

    start_time = st.time_input("Start Time", value=min_time)
    end_time = st.time_input("End Time", value=max_time)

    if start_time and end_time:
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

def main():
    """Main function to run the Streamlit dashboard."""
    charts, perf_charts, metrics_placeholders = initialize_ui()
    
    df = load_data(DATA_FILE)
    if df is None:
        st.stop()

    col_updated, initial_message = display_static_info(df)
    slider_placeholder = display_voltage_threshold_info(df)
    fetch_weather_data(CITIES)
    display_system_performance()

    # --- AFO Setup ---
    st.markdown("### Automatic Optimization (Live)")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    afo_placeholder = st.empty()
    if numeric_columns:
        target_column = st.selectbox("Select target column for optimization:", options=numeric_columns, index=0)
        k = st.slider("Number of features to select", min_value=1, max_value=len(numeric_columns)-1, value=min(3, len(numeric_columns)-1))
    else:
        st.warning("No numeric columns available for optimization.")
        target_column = None
        k = 0

    # --- Real-time Loop ---
    all_updated_columns = {}
    live_data_buffer = []
    
    # Use df.to_dict('records') for a slightly faster iteration than iterrows
    for i, row in enumerate(df.to_dict('records')):
        if "Voltage" in row and pd.notna(row["Voltage"]):
            slider_placeholder.slider(
                "Voltage Threshold (Live)",
                min_value=0.0,
                max_value=300.0,
                step=0.1,
                value=float(row["Voltage"]),
                key="voltage_slider"
            )
        
        update_performance_charts(perf_charts)
        updated_columns = update_real_time_charts(charts, row)
        all_updated_columns.update(updated_columns)
        
        live_data_buffer.append(row)
        
        # Update live AFO every 10 rows
        if (i + 1) % 10 == 0 and target_column:
            df_live = pd.DataFrame(live_data_buffer)
            result_df, selected_features = automatic_feature_optimization(df_live, target_column, k)
            if result_df is not None:
                with afo_placeholder.container():
                    st.markdown("#### Live Automatic Optimization Results")
                    st.dataframe(result_df)
                    st.write("Selected Features:", selected_features)
        
        if "Date" in row:
            metrics_placeholders["date"].metric("Date", row["Date"])
        if "Time" in row:
            # Format time to string to avoid issues with datetime objects
            time_str = row["Time"].strftime('%H:%M:%S') if hasattr(row["Time"], 'strftime') else str(row["Time"])
            metrics_placeholders["time"].metric("Time", time_str)
        
        time.sleep(0.1)

    # --- Final Display ---
    initial_message.empty()
    with col_updated.container():
        st.markdown("### Updated Columns with Last Known Values:")
        formatted_updated_columns = [
            {"Column": col, "Last Known Value": value}
            for col, value in all_updated_columns.items()
            if col in RELEVANT_COLUMNS and pd.notna(value)
        ]
        if formatted_updated_columns:
            st.table(pd.DataFrame(formatted_updated_columns))

    display_historical_data(df)

if __name__ == "__main__":
    main()