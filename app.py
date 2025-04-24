import streamlit as st
import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# Streamlit UI configuration
st.set_page_config(page_title="ScentMatch", layout="wide")
st.title("🌸 ScentMatch – Discover Your Perfect Perfume")
st.markdown("Select a perfume from the list to find similar fragrances based on accords and gender.")

# Configuration
USE_DESCRIPTION = False  # Exclude Description to reduce memory usage
MAX_FEATURES = 50  # Minimal features to avoid memory issues

# Load and cache the dataset
@st.cache_data
def load_data():
    try:
        if not os.path.exists('fra_perfumes.csv'):
            st.error("Error: 'fra_perfumes.csv' file not found. Please ensure the file is in the same directory as this script.")
            return None
        df = pd.read_csv('fra_perfumes.csv', encoding='utf-8')
        
        # Clean and preprocess data
        df.fillna({
            'Name': '',
            'Gender': 'Unknown',
            'Main Accords': '[]',
            'Description': '',
            'Rating Value': 0,
            'Rating Count': 0,
            'url': ''
        }, inplace=True)
        
        # Convert all relevant columns to strings
        for col in ['Name', 'Gender', 'Description']:
            df[col] = df[col].astype(str)
        
        # Convert Main Accords to string to avoid hashing issues
        def parse_accords_to_string(x):
            try:
                accords = ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else []
                return ' '.join(accords) if isinstance(accords, list) else ''
            except:
                return ''
        df['Main Accords'] = df['Main Accords'].apply(parse_accords_to_string)
        
        # Create Accords_Str for consistency
        df['Accords_Str'] = df['Main Accords']
        
        # Combine features for TF-IDF
        df['Combined_Features'] = (
            df['Name'] + ' ' +
            df['Gender'] + ' ' +
            df['Accords_Str']
        )
        
        # Ensure Combined_Features is a valid string
        df['Combined_Features'] = df['Combined_Features'].astype(str).replace('', 'unknown')
        
        # Check for invalid rows
        invalid_rows = df[df['Combined_Features'].isna() | (df['Combined_Features'] == '')]
        if not invalid_rows.empty:
            st.warning(f"Found {len(invalid_rows)} rows with invalid Combined_Features. These will be filled with 'unknown'.")
            df.loc[invalid_rows.index, 'Combined_Features'] = 'unknown'
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Compute TF-IDF and cache vectorizer
@st.cache_resource
def compute_tfidf(df):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
        tfidf_matrix = vectorizer.fit_transform(df['Combined_Features'])
        return tfidf_matrix, vectorizer
    except Exception as e:
        st.error(f"Error computing TF-IDF: {str(e)}")
        return None, None

# Get recommendations
def get_recommendations(perfume_name, df, tfidf_matrix, vectorizer, top_n=5):
    try:
        indices = pd.Series(df.index, index=df['Name'].str.lower().str.strip())
        idx = indices.get(perfume_name.lower().strip())
        
        if idx is None:
            return "Perfume not found in the dataset."
        
        # Compute similarity for the single perfume
        perfume_vector = tfidf_matrix[idx]
        sim_scores = cosine_similarity(perfume_vector, tfidf_matrix).flatten()
        
        # Get top N similar perfumes
        sim_indices = sim_scores.argsort()[::-1][1:top_n+1]  # Skip the input perfume
        sim_scores = sim_scores[sim_indices]
        
        # Prepare results
        results = df.iloc[sim_indices][['Name', 'Gender', 'Rating Value', 'Rating Count', 'Main Accords', 'Description', 'url']].copy()
        results['Similarity'] = sim_scores
        
        # Convert Main Accords back to list for display
        results['Main Accords'] = results['Main Accords'].apply(lambda x: x.split() if x else [])
        
        return results
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

# Main app logic
df = load_data()
if df is None:
    st.stop()

tfidf_matrix, vectorizer = compute_tfidf(df)
if tfidf_matrix is None:
    st.stop()

# User input via dropdown
st.subheader("Select a Perfume")
perfume_names = ['Select a perfume'] + sorted(df['Name'].str.strip().tolist())  # Sort alphabetically
user_input = st.selectbox("Choose a perfume you love:", perfume_names, index=0)

if user_input == 'Select a perfume':
    st.info("Please select a perfume to see recommendations.")
else:
    with st.spinner("Finding similar perfumes..."):
        results = get_recommendations(user_input, df, tfidf_matrix, vectorizer)
        
        if isinstance(results, str):
            st.warning(results)
        else:
            st.subheader("🌟 Recommended Perfumes")
            for _, row in results.iterrows():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{row['Name']}** (Similarity: {row['Similarity']:.2f})")
                    st.write(f"**Gender**: {row['Gender']}")
                    st.write(f"**Rating**: {row['Rating Value']:.2f} ({row['Rating Count']} reviews)")
                    st.write(f"**Main Accords**: {', '.join(row['Main Accords'][:5])}")
                    st.write(f"**Description**: {row['Description'][:200]}...")
                    st.markdown(f"[Learn more on Fragrantica]({row['url']})")
                with col2:
                    st.write("")  # Placeholder for potential image
                st.markdown("---")

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa;
    padding: 10px;
    text-align: center;
    font-size: 12px;
    color: #6c757d;
}
</style>
<div class="footer">
    Powered by ScentMatch | Data sourced from Fragrantica
</div>
""", unsafe_allow_html=True)