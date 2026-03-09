import streamlit as st
import pandas as pd

from src.data_preprocessing import load_and_clean_data
from src.recommendation import build_models, recommend_books, cluster_recommend

st.title("📚 Book Recommendation System")

st.write("Discover books similar to your favorite reads.")

# Load data
df = load_and_clean_data()

cosine_sim, df = build_models(df)

# Use lowercase column names
book_list = df["book name"].values

selected_book = st.selectbox(
    "Choose a book you like",
    book_list
)

model_type = st.selectbox(
    "Recommendation Method",
    ["Content Based", "Cluster Based"]
)

if st.button("Recommend Books"):

    if model_type == "Content Based":
        recs = recommend_books(selected_book, df, cosine_sim)
    else:
        recs = cluster_recommend(selected_book, df)

    st.subheader("Recommended Books:")

    for i in recs:
        st.write(i)

st.subheader("Dataset Insights")

st.write("Total Books:", len(df))

# Fix lowercase column
st.write("Average Rating:", round(df["rating"].mean(), 2))

# Fix lowercase column
top_authors = df["author"].value_counts().head(5)

st.bar_chart(top_authors)