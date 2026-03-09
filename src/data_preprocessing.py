import pandas as pd

def load_and_clean_data():

    df1 = pd.read_csv("Audible_Catlog.csv")
    df2 = pd.read_csv("Audible_Catlog_Advanced_Features.csv")

    # Convert column names to lowercase
    df1.columns = df1.columns.str.lower()
    df2.columns = df2.columns.str.lower()

    # Merge datasets
    df = pd.merge(df1, df2, on=["book name", "author"], how="left")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fix duplicate columns safely
    if "rating_x" in df.columns and "rating_y" in df.columns:
        df["rating"] = df["rating_x"].fillna(df["rating_y"])

    if "number of reviews_x" in df.columns and "number of reviews_y" in df.columns:
        df["number of reviews"] = df["number of reviews_x"].fillna(df["number of reviews_y"])

    if "price_x" in df.columns and "price_y" in df.columns:
        df["price"] = df["price_x"].fillna(df["price_y"])

    # Drop duplicate columns safely
    df.drop(columns=[
        "rating_x", "rating_y",
        "number of reviews_x", "number of reviews_y",
        "price_x", "price_y"
    ], inplace=True, errors="ignore")

    # Handle missing values safely
    if "rating" in df.columns:
        df["rating"] = df["rating"].fillna(df["rating"].mean())

    if "description" in df.columns:
        df["description"] = df["description"].fillna("No description available")

    # Convert datatypes safely
    if "number of reviews" in df.columns:
        df["number of reviews"] = pd.to_numeric(df["number of reviews"], errors="coerce")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Fill remaining NaN values
    df.fillna(0, inplace=True)

    return df
