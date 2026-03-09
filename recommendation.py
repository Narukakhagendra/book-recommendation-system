import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_models(df):

    # make sure description column exists
    if "description" not in df.columns:
        raise ValueError("description column not found in dataset")

    # fill missing values
    df["description"] = df["description"].fillna("")

    # TF-IDF model
    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(df["description"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim, df


def recommend_books(book_name, df, cosine_sim, top_n=5):

    # find book index
    idx = df[df["book name"] == book_name].index[0]

    # similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # top recommendations
    sim_scores = sim_scores[1:top_n+1]

    book_indices = [i[0] for i in sim_scores]

    return df["book name"].iloc[book_indices].tolist()


def cluster_recommend(book_name, df):

    # simple fallback recommendation
    same_author = df[df["author"] == df[df["book name"] == book_name]["author"].values[0]]

    return same_author["book name"].head(5).tolist()