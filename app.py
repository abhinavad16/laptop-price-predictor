import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained pipeline and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("💻 Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Laptop Type
type= st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop (in KG)', min_value=0.1, value=1.0, step=0.1, format="%.2f")

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS Display', ['No', 'Yes'])

# Screen size
screensize = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.3)

# Screen Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):
    # Convert Touchscreen and IPS to binary
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screensize

    input_df = pd.DataFrame([[company, type, ram, weight, touchscreen, ips, ppi,
                              cpu, hdd, ssd, gpu, os]],
                            columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
                                     'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])


    prediction = pipe.predict(input_df)[0]
    st.title("Laptop Price: ₹ " + str(int(np.exp(prediction))))
