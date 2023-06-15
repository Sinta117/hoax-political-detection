import streamlit as st
import pandas as pd
import numpy as np
import json

import detekfix_hoax

st.write("""
# Fake Political News Detection
""")

w2i, i2w = {'Hoax': 0, 'Non Hoax': 1}, {0: 'Hoax', 1: 'Non Hoax'}

input = st.text_input('Teks Judul Berita Politik', '')

col1, col2, col3 = st.columns(3)
push = col3.button("Detection")

disp1, disp2, disp3 = st.columns(3)

if push:
    label_RF = detekfix_hoax.predict_model_rf(input)
    if label_RF[0].argmax()==0:
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(255, 0, 0);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: red;
            font-size:200%;
            }
            </style>
            """
            , unsafe_allow_html=True)

    else:
        st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1% 1% 1% 1%;
            border-radius: 2px;
            color: rgb(13, 252, 13);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: green;
            font-size:200%;
            }
            </style>
            """
            , unsafe_allow_html=True)

    score = label_RF[0].max()

    disp2.metric(i2w[label_RF[0].argmax()], '{}%'.format(score*100))

