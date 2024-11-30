import streamlit as st
import time
import pandas as pd
import requests
import psutil
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(
    page_title='Smart Dashboard',
    layout="wide",
)

# Init kolona za dijagrame
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

# Definisanje subheader-a za dijagrame
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

# Prikaz Datuma i Vremena
YMDt = st.empty()
Tt = st.empty()

# Ucitavanje Dataset-a i Error handling
try:
    df = pd.read_csv("demo_data.csv")
except FileNotFoundError:
    st.error("Data file 'demo_data.csv' not found. Please ensure it is in the correct directory.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("The data file 'demo_data.csv' is empty. Please provide a valid CSV file with data.")
    st.stop()
except pd.errors.ParserError:
    st.error("Error while parsing the CSV file. Please ensure the file format is correct.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the data: {e}")
    st.stop()

# prikaz kolona iz dataset-a i placeholder za azurirane kolone u istom redu
with st.container():
    col_dataset, col_updated = st.columns(2)
    
    # Prikaz dataset kolona 
    col_dataset.markdown("### Columns in dataset:")
    dataset_columns_df = pd.DataFrame(df.columns.tolist(), columns=["Columns"])
    col_dataset.table(dataset_columns_df)

    # Prikaz placeholder-a za tabelu azuriranih kolona
    initial_message = col_updated.info("The list of updated columns with their last known values will be displayed here after processing is complete.")

# Filtriranje kolona u dataset-u
st.markdown("### Filter Columns")
selected_columns = st.multiselect("Select columns to display:", options=df.columns)
if selected_columns:
    st.write("Filtered Columns Data:")
    st.table(df[selected_columns])

# Korisnički definisani threshold-ovi
st.markdown("### Voltage Threshold")
voltage_threshold = st.slider("Set Voltage Threshold", min_value=0.0, max_value=300.0, step=0.1)
if df["Voltage"].max() > voltage_threshold:
    st.warning("Voltage exceeded threshold!")

# Sumiranje podataka
st.markdown("### Dataset Summary")
st.write(df.describe())

# Integracija sa eksternalnim API
st.markdown("### Weather Data")
try:
    weather = requests.get("https://api.open-meteo.com/v1/forecast?latitude=35&longitude=139&hourly=temperature_2m").json()
    st.write("Temperature (°C):", weather["hourly"]["temperature_2m"][:5])
except Exception as e:
    st.error(f"Unable to fetch weather data: {e}")

# Pracenje azuriranih kolona
def update_charts(data):
    updated_columns = {}

    try:
        if "Voltage" in data and "Current" in data:
            Voltage.add_rows([{"Voltage": data["Voltage"], "Current": data["Current"]}])
            updated_columns["Voltage"] = data["Voltage"]
            updated_columns["Current"] = data["Current"]

        if "Measured_Frequency" in data:
            Freq.add_rows([{"Measured_Frequency": data["Measured_Frequency"]}])
            updated_columns["Measured_Frequency"] = data["Measured_Frequency"]

        if all(k in data for k in ["Active_Power", "Reactive_Power", "Apperent_Power"]):
            AP.add_rows([{
                "Active_Power": data["Active_Power"],
                "Reactive_Power": data["Reactive_Power"],
                "Apperent_Power": data["Apperent_Power"]
            }])
            updated_columns["Active_Power"] = data["Active_Power"]
            updated_columns["Reactive_Power"] = data["Reactive_Power"]
            updated_columns["Apperent_Power"] = data["Apperent_Power"]

        if "Phase_Voltage_Angle" in data:
            PA.add_rows([{"Phase_Voltage_Angle": data["Phase_Voltage_Angle"]}])
            updated_columns["Phase_Voltage_Angle"] = data["Phase_Voltage_Angle"]

        if "Cos_Phi" in data:
            CP.add_rows([{"Cos_Phi": data["Cos_Phi"]}])
            updated_columns["Cos_Phi"] = data["Cos_Phi"]

        if "Power_Factor" in data:
            PF.add_rows([{"Power_Factor": data["Power_Factor"]}])
            updated_columns["Power_Factor"] = data["Power_Factor"]
    except Exception as e:
        st.error(f"Error while updating charts: {e}")
    
    return updated_columns

# Iteracija kroz redove DataFrame-a
all_updated_columns = {}
for i, row in df.iterrows():
    try:
        if "Date" in row:
            YMDt.metric("Date", row["Date"])
        if "Time" in row:
            Tt.metric("Time", row["Time"])

        updated_columns = update_charts(row)
        all_updated_columns.update(updated_columns)
        
        time.sleep(0.01)
    except KeyError as e:
        st.warning(f"Missing expected column: {e}")
    except Exception as e:
        st.error(f"An error occurred while processing row {i}: {e}")

# Prikaz azuriranih kolona nakon obrade
try:
    initial_message.empty()

    with st.container():
        updated_columns_header = col_updated.markdown("### Updated Columns with Last Known Values:")
        formatted_updated_columns = [{"Column": col, "Last Known Value": value} for col, value in all_updated_columns.items()]
        updated_columns_df = pd.DataFrame(formatted_updated_columns)
        col_updated.table(updated_columns_df)
except Exception as e:
    st.error(f"An error occurred while displaying updated columns: {e}")

# Napredno praćenje performansi sistema
st.markdown("### Performance monitor")
with st.expander("System performance overview"):
    # Prikaz osnovnih performansi
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()

    # Prikaz detaljnih podataka
    st.write(f"**CPU usage:** {cpu_usage}%")
    st.write(f"**RAM usage:** {memory_info.percent}%")
    st.write(f"**Total RAM memory:** {memory_info.total / (1024 ** 3):.2f} GB")
    st.write(f"**Available RAM memory:** {memory_info.available / (1024 ** 3):.2f} GB")
    st.write(f"**Disk usage:** {disk_usage.percent}%")
    st.write(f"**Total space on the Disk:** {disk_usage.total / (1024 ** 3):.2f} GB")
    st.write(f"**Available memory on Disk:** {disk_usage.free / (1024 ** 3):.2f} GB")
    st.write(f"**Sent data:** {net_io.bytes_sent / (1024 ** 2):.2f} MB")
    st.write(f"**Data received:** {net_io.bytes_recv / (1024 ** 2):.2f} MB")

# Dodavanje grafova za CPU, RAM i Disk
st.markdown("#### Real-time Monitoring for CPU, RAM and Disk")
perf_col1, perf_col2, perf_col3 = st.columns(3)
cpu_chart = perf_col1.line_chart({"CPU Usage": []})
memory_chart = perf_col2.line_chart({"Memory Usage": []})
disk_chart = perf_col3.line_chart({"Disk Usage": []})

# Real-time praćenje podataka
for _ in range(100):
    cpu_chart.add_rows({"CPU Usage": [psutil.cpu_percent()]})
    memory_chart.add_rows({"Memory Usage": [psutil.virtual_memory().percent]})
    disk_chart.add_rows({"Disk Usage": [psutil.disk_usage('/').percent]})
    time.sleep(0.5)

# PDF Generacija izvestaja
def generate_pdf(report_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont