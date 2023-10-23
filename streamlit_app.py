# Imports required
import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.write("""
# Bizmatch Tag System

Generate tag groups to match a company to a wider network.

""")

with st.expander("Follow the steps below to generate tag groupings:"):
  st.write(
      """1. Upload a .csv file of tags (industries).
            \n2. Click the button **Group Tags** to generate tag groupings.
            \n3. View the .csv file of tag groupings.
            \n4. Download the .csv file of tag groupings by clicking the **Download** button.
            \n5. View the network graph to visualize the potential matches for a particular tag.
            """)

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
if uploaded_file is not None:
  # Can be used wherever a "file-like" object is accepted:
  tags_df = pd.read_csv(uploaded_file)
  st.dataframe(tags_df)
else:
  tags = []

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the tags into TF-IDF vectors
tags = tags_df.values.tolist()
tfidf_matrix = tfidf_vectorizer.fit_transform(tags)

# Calculate cosine similarity between all tag pairs
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define a threshold for considering tags similar
threshold = 0.5

# Create a dictionary to store groups of similar tags
tag_groups = {}

# Iterate through the tags and find similar tags
for i, tag in enumerate(tags):
    similar_tags = [tags[j] for j in range(len(tags)) if cosine_sim[i][j] > threshold]
    tag_groups[tag] = similar_tags

# Print the tag groups
for tag, similar_tags in tag_groups.items():
    st.write(tag, ":", similar_tags)

# Enable webview in replit
os.system("streamlit run streamlit_app.py --server.enableCORS false")