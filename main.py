# Imports required
import streamlit as st
import os
import pandas as pd
import numpy as np

st.write("""
# Bizmatch Tag System

Generate tag groups to match a company to a wider network.

""")

with st.expander("Follow the steps below to generate tag groupings:"):
  st.write(
      """1. Upload a .csv file of tags (industries) OR paste a list of tags, separated by a new line, into the text box.
            \n2. Click the button **Group Tags** to generate tag groupings.
            \n3. View the .csv file of tag groupings.
            \n4. Download the .csv file of tag groupings by clicking the **Download** button.
            \n5. View the network graph to visualize the potential matches for a particular tag.
            """)

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
if uploaded_file is not None:
  # Can be used wherever a "file-like" object is accepted:
  dataframe = pd.read_csv(uploaded_file)
  st.write(dataframe)
  st.dataframe(dataframe)

# Text Input
st.subheader('st.text_area')
input_tags = st.text_area('Paste tags here')
tags = input_tags.split("\n")
# output
st.write(input_tags)
#tags_df = pd.DataFrame(input_tags)
#st.dataframe(tags_df)

# Enable webview in replit
os.system("streamlit run main.py --server.enableCORS false")
