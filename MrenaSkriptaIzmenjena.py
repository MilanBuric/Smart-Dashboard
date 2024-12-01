import streamlit as st
import time
import numpy as np
import pandas as pd

st.set_page_config(
    page_title='Live DashBoard',
    layout="wide",
)

df = pd.read_csv("demo_data.csv")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Voltage[V] and Current\n")
    Voltage = st.line_chart()

with col2:
    st.subheader("Frequency\n")
    Freq = st.line_chart()

with col3:
    st.subheader("Power Data\n")
    AP = st.line_chart()

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Angle Data\n")
    PA = st.line_chart()

with col5:
    st.subheader("Cos Phi Data\n")
    CP = st.line_chart()

with col6:
    st.subheader("Power Factor Data\n")
    PF = st.line_chart()

isim = ":"
col7 = st.columns(1)[0]
col7.write(f"Experiment Data \n {isim}")
YMDt = col7.empty()
Tt = col7.empty()

for i in range(len(df)):
    buff = df.iloc[i]

    with Tt:
        st.metric("Date", buff[0])
    with YMDt:
        st.metric("Time", buff[1])
    
    Frequency_data = {
        "Measured Frequency": float(buff[10])
    }
    Freq.add_rows([Frequency_data])
    
    voltage_data = {
        "Voltage": float(buff[8]),
        "Current": float(buff[9])
    }
    Voltage.add_rows([voltage_data])

    Power_data = {
        "Active Power": float(buff[10]),
        "Reactive Power": float(buff[11]),
        "Apparent Power": float(buff[12]),
    } 
    AP.add_rows([Power_data])

    Phase_Angle_data = {
        "Phase Voltage Angle": float(buff[13]),
        "Reactive Current Power": float(buff[14])
    }
    PA.add_rows([Phase_Angle_data])

    COS_PHI_data = {
        "COS PHI": float(buff[15])
    }
    CP.add_rows([COS_PHI_data])

    Power_Factor_data = {
        "COS PHI": float(buff[16])
    }
    PF.add_rows([Power_Factor_data])
    
    time.sleep(0.2)
