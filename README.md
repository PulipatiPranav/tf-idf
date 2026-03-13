# TF-IDF Similarity Matching System

## Project Overview
This repository contains the implementation of a Term Frequency-Inverse Document Frequency (TF-IDF) similarity matching system designed for patent evaluation. This system was developed during my internship at Eiger Tech, aimed at improving the efficiency and accuracy of patent searches through effective text analysis.

## Technical Architecture
The architecture of the TF-IDF system is built on a combination of the following components:
1. **Data Ingestion**: Collects and preprocesses textual data from various patent databases.
2. **TF-IDF Computation**: Calculates the TF-IDF scores for each term in the datasets, allowing for the evaluation of term importance across the documents.
3. **Similarity Matching**: Utilizes cosine similarity measures to evaluate the similarity between documents based on their TF-IDF vectors.
4. **User Interface (UI)**: A simple interface for users to input queries and view matching patents ranked by relevance.

## Usage Instructions
To use the TF-IDF similarity matching system, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/PulipatiPranav/tf-idf.git
   cd tf-idf
   ```
2. Install necessary dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```
4. Enter a query in the provided input field to fetch matching patents.

## Algorithm Explanation
The core algorithm leverages the TF-IDF model, which operates through the following stages:
- **Tokenization**: Splittling the text into individual terms or tokens.
- **Term Frequency (TF)**: Calculating the frequency of each term in a document.
- **Inverse Document Frequency (IDF)**: Evaluating the rarity of a term across the entire set of documents.
- **TF-IDF Score**: Combining TF and IDF scores to determine term importance:
  \[ TFIDF(t, d) = TF(t, d) \times IDF(t) \]
- **Cosine Similarity**: Measuring the distance between document vectors represented in TF-IDF space.
  \[ CosineSimilarity(A, B) = \frac{A \cdot B}{||A|| \times ||B||} \]

## Future Enhancements
Future developments to enhance the functionality of this system could include:
- **Integration of Machine Learning (ML)** techniques to improve the accuracy of matches based on historical querying data.
- **User Feedback Loop** to refine the search parameters and optimize results based on user interactions.
- **Advanced UI Features** such as dynamic filtering and sorting of results to improve user experience.

---
This readme serves as a comprehensive guide to the TF-IDF Similarity matching system for patent evaluation. For any issues or contributions, please feel free to open a pull request or raise an issue in the repository.