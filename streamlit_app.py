# Imports required
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
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
col1, col2, col3 = st.columns(3)

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
    st.write("Generated Tag Groups (TF-IDF vectorizer)")
    st.write(tag_groups_df)

  ######################################################################
  # GENSIM Application
  # Preprocess the tags: split into words and make them lowercase
  tag_tokens = [tag.lower().split() for tag in tags]

  # Create and train a Word2Vec model
  model = Word2Vec(sentences=tag_tokens, vector_size=100, window=5, min_count=1, sg=0)

  # Define a function to calculate the vector representation of a tag
  def get_tag_vector(tag):
      tokens = tag.lower().split()
      vector = np.zeros(model.vector_size)
      for token in tokens:
          if token in model.wv:
              vector += model.wv[token]
      return vector

  # Calculate cosine similarity between all tag pairs
  cosine_sim_gensim = cosine_similarity([get_tag_vector(tag) for tag in tags])

  # Define a threshold for considering tags similar
  threshold_gensim = 0.5

  # Create a dictionary to store groups of similar tags
  tag_groups_gensim = {}

  # Iterate through the tags and find similar tags
  for i, tag in enumerate(tags):
      similar_tags_gensim = [tags[j] for j in range(len(tags)) if cosine_sim_gensim[i][j] > threshold_gensim]
      tag_groups_gensim[tag] = similar_tags_gensim

  tag_groups_gensim_df = pd.DataFrame.from_dict(tag_groups_gensim, orient='index')
  # Visualize generated tag groups
  with col3:
    st.write("Generated Tag Groups (Gensim-Word2Vec)")
    st.write(tag_groups_gensim_df)
  # Print the tag groups
  #for tag, similar_tags_gensim in tag_groups_gensim.items():
      #print(tag, ":", similar_tags_gensim)

  ######################################################################
  # Download button to download the csv file
  @st.cache_data
  def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

  csv_tfidf = convert_df(tag_groups_df)
  csv_gensim = convert_df(tag_groups_gensim_df)

  with col2:
    if st.download_button(
        label="Download",
        type="primary",
        data=csv_tfidf,
        file_name='tag_groups_tfidf.csv',
        mime='text/csv',
    ):
      st.write('File Downloaded')
    else:
      st.write('')

    with col3:
      if st.download_button(
          label="Download",
          type="primary",
          data=csv_gensim,
          file_name='tag_groups_gensim.csv',
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

    # Columns for printing
    col_tfidf, col_gensim= st.columns(2)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
      with col_tfidf:
        st.write("TF-IDF vectorizer")
      with col_gensim:
        st.write("Gensim-Word2Vec")
      for tag, similar_tags in tag_groups.items():
        for option in options:
          if option == tag:
            with col_tfidf:
              st.write(tag, ":", similar_tags)
      for tag, similar_tags in tag_groups_gensim.items():
        for option in options:
          if option == tag:
            with col_gensim:
              st.write(tag, ":", similar_tags)
    else:
      st.write('')
