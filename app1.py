import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_tfidf_similarity(source_text, reference_texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([source_text] + reference_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

st.markdown(
    """
    <style>
    .stApp h1 {
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Top Ten Matches (TF-IDF)")

source_file = st.file_uploader("Upload Source Excel File", type=["csv", "xlsx"])
reference_file = st.file_uploader("Upload Reference Excel File", type=["csv", "xlsx"])

columns_to_compare = ["Title", "Abstract"]

if source_file and reference_file:
    source_df = pd.read_excel(source_file) if source_file.name.endswith('.xlsx') else pd.read_csv(source_file)
    reference_df = pd.read_excel(reference_file) if reference_file.name.endswith('.xlsx') else pd.read_csv(reference_file)

    source_df.fillna("", inplace=True)
    reference_df.fillna("", inplace=True)

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

        top_matches = reference_df.nlargest(10, f"{columns_to_compare[0]}_Similarity")

        for column in columns_to_compare[1:]:
            column_top_matches = reference_df.nlargest(10, f"{column}_Similarity")
            top_matches = pd.concat([top_matches, column_top_matches]).drop_duplicates()

        st.subheader("Top 10 Similarity Results")
        st.write(top_matches)

        csv = top_matches.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Top 10 Matches as CSV",
            data=csv,
            file_name="top_10_similarity_results.csv",
            mime="text/csv",
        )
