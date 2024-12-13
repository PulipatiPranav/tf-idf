import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_tfidf_similarity(source_text, reference_texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([source_text] + reference_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

st.set_page_config(page_title="Top Ten Matches (TF-IDF)", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --primary-bg-light: #f9f9f9;
        --primary-bg-dark: #0e1117;
        --primary-text-light: #000000;
        --primary-text-dark: #ffffff;
        --secondary-bg-light: #ffffff;
        --secondary-bg-dark: #1c1f26;
        --secondary-text-light: #333333;
        --secondary-text-dark: #dddddd;
        --border-color-light: #dddddd;
        --border-color-dark: #444444;
    }
    html[data-theme="light"] .stApp {
        background-color: var(--primary-bg-light);
        color: var(--primary-text-light);
    }
    html[data-theme="dark"] .stApp {
        background-color: var(--primary-bg-dark);
        color: var(--primary-text-dark);
    }
    html[data-theme="light"] .stFileUploader label,
    html[data-theme="dark"] .stFileUploader label {
        background-color: var(--secondary-bg-light);
        color: var(--secondary-text-light);
        border: 1px solid var(--border-color-light);
    }
    html[data-theme="dark"] .stFileUploader label {
        background-color: var(--secondary-bg-dark);
        color: var(--secondary-text-dark);
        border: 1px solid var(--border-color-dark);
    }
    h1 {
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Top Ten Matches (TF-IDF)")

source_file = st.file_uploader("Upload Source Excel File", type=["csv", "xlsx"])
reference_file = st.file_uploader("Upload Reference Excel File", type=["csv", "xlsx"])

comparison_option = st.radio(
    "Choose comparison criteria:",
    ["Title", "Abstract", "All of the Above"],
    index=2
)

if source_file and reference_file:
    source_df = pd.read_excel(source_file) if source_file.name.endswith('.xlsx') else pd.read_csv(source_file)
    reference_df = pd.read_excel(reference_file) if reference_file.name.endswith('.xlsx') else pd.read_csv(reference_file)
    source_df.fillna("", inplace=True)
    reference_df.fillna("", inplace=True)

    columns_to_compare = []
    if comparison_option == "Title":
        columns_to_compare = ["Title"]
    elif comparison_option == "Abstract":
        columns_to_compare = ["Abstract"]
    elif comparison_option == "All of the Above":
        columns_to_compare = ["Title", "Abstract"]

    missing_columns = [col for col in columns_to_compare if col not in source_df.columns or col not in reference_df.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)} in the uploaded files.")
    else:
        output = pd.DataFrame()
        for column in columns_to_compare:
            source_text = source_df[column].iloc[0]
            reference_texts = reference_df[column].tolist()
            similarities = compute_tfidf_similarity(source_text, reference_texts)
            reference_df[f"{column}_Similarity"] = similarities

        if len(columns_to_compare) > 1:
            reference_df["Combined_Similarity"] = reference_df[[f"{col}_Similarity" for col in columns_to_compare]].mean(axis=1)
            top_matches = reference_df.nlargest(10, "Combined_Similarity")
        else:
            top_matches = reference_df.nlargest(10, f"{columns_to_compare[0]}_Similarity")

        st.subheader("Top 10 Similarity Results")
        st.write(top_matches)

        csv = top_matches.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Top 10 Matches as CSV",
            data=csv,
            file_name="top_10_similarity_results.csv",
            mime="text/csv",
        )
