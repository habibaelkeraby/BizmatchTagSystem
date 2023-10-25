# Imports required
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

######################################################################
st.write("""
# Bizmatch Tag System

Generate tag groups to match a company to a wider network.

""")
######################################################################
# Expandable section for general instructions
with st.expander("Follow the steps below to generate tag groupings:"):
  st.write("""1. Upload a .csv file of tags (industries).
            \n2. Once the file is uploaded, you will be able to view the uploaded data, as well as the generated tag groupings.
            \n3. You can download the .csv file of tag groupings by clicking the **Download** button.
            \n4. You can visualize the potential matches for a particular tag or tags in the multi-select widget.
            """)

######################################################################
# File Uploader
st.write("## File Uploader")
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

# Expandable section for csv guidelines
with st.expander("Guidelines on formatting of csv file:"):
  st.write(
      """1. The csv file should contain a single column, which can have a column header.
            \n2. Each entry should correspond to one tag, i.e., one industry.
            \n3. The contents of the file must be in English.
            """)
# Column structure to visualize data
col1, col2 = st.columns(2)

# Processing uploaded file
if uploaded_file is not None:
  # Can be used wherever a "file-like" object is accepted:
  tags_df = pd.read_csv(uploaded_file)
  # Visualize uploaded file
  with col1:
    st.write("Uploaded File")
    st.write(tags_df)
  tags = tags_df[tags_df.columns.values[0]].tolist()

  # Creating tag groupings
  # Create a TF-IDF vectorizer
  tfidf_vectorizer = TfidfVectorizer()

  # Fit and transform the tags into TF-IDF vectors
  tfidf_matrix = tfidf_vectorizer.fit_transform(tags)

  # Calculate cosine similarity between all tag pairs
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

  # Define a threshold for considering tags similar
  threshold = 0.5

  # Create a dictionary to store groups of similar tags
  tag_groups = {}

  # Iterate through the tags and find similar tags
  for i, tag in enumerate(tags):
    similar_tags = [
        tags[j] for j in range(len(tags)) if cosine_sim[i][j] > threshold
    ]
    tag_groups[tag] = similar_tags

  #tag_groups_df = pd.DataFrame([(k,pd.Series(v)) for k,v in tag_groups.items()])
  tag_groups_df = pd.DataFrame.from_dict(tag_groups, orient='index')
  # Visualize generated tag groups
  with col2:
    st.write("Generated Tag Groups")
    st.write(tag_groups_df)

  ######################################################################
  # Download button to download the csv file
  @st.cache_data
  def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

  csv = convert_df(tag_groups_df)

  if st.download_button(
      label="Download",
      type="primary",
      data=csv,
      file_name='tag_groups.csv',
      mime='text/csv',
  ):
    st.write('File Downloaded')
  else:
    st.write('')

    # Print the tag groups
    #for tag, similar_tags in tag_groups.items():
    #  st.write(tag, ":", similar_tags)

  ######################################################################
  # Multi-Select widget to visualize similar tags
  with st.form("my_form"):
    st.write("## Multi-Select Widget")
    options = st.multiselect(
        'Select tag(s) to check the tags they will be matched to:', tags)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
      for tag, similar_tags in tag_groups.items():
        for option in options:
          if option == tag:
            st.write(tag, ":", similar_tags)
    else:
      st.write('')
