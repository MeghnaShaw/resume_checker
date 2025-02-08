import streamlit as st
import PyPDF2
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import nltk
import os
import tempfile
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK data is available
nltk.download('punkt')

st.set_page_config(page_title="AI Hiring Assistant", layout="wide")

st.title("AI Hiring Assistant")
st.markdown("Upload resumes and a job description to find the best-matched candidate.")

# Upload resumes
uploaded_resumes = st.file_uploader("Upload Resume PDFs", type="pdf", accept_multiple_files=True)
# Upload job description
uploaded_job_desc = st.file_uploader("Upload Job Description PDF", type="pdf")

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name
    
    with open(temp_pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    
    return text.strip()

if uploaded_resumes and uploaded_job_desc:
    st.success("Files uploaded successfully!")
    
    # Extract job description text
    job_desc_text = extract_text_from_pdf(uploaded_job_desc)
    
    if job_desc_text:
        st.subheader("Job Description Text")
        st.text_area("Job Description", job_desc_text, height=150)
        
        # Extract resume texts and store names
        resume_texts = {}
        for resume in uploaded_resumes:
            text = extract_text_from_pdf(resume)
            if text:
                resume_texts[resume.name] = text
        
        if resume_texts:
            # Tokenize and preprocess text
            all_texts = [job_desc_text] + list(resume_texts.values())
            sentences = [sent_tokenize(text) for text in all_texts]
            preprocessed_sentences = [simple_preprocess(sentence) for text in sentences for sentence in text if sentence.strip()]
            
            if preprocessed_sentences:
                # Train Word2Vec model
                model = Word2Vec(sentences=preprocessed_sentences, vector_size=100, window=5, min_count=1, workers=4)
                
                # Compute similarity scores
                def get_embedding(text):
                    words = simple_preprocess(text)
                    vectors = [model.wv[word] for word in words if word in model.wv]
                    return np.mean(vectors, axis=0) if vectors else np.zeros(100)
                
                job_vector = get_embedding(job_desc_text)
                resume_scores = {name: cosine_similarity([job_vector], [get_embedding(text)])[0][0] for name, text in resume_texts.items()}
                
                # Find best-matched candidate
                best_candidate = max(resume_scores, key=resume_scores.get)
                
                st.subheader("Best Matched Candidate")
                st.success(f"**{best_candidate}** is the best-matched candidate with a similarity score of {resume_scores[best_candidate]:.4f}!")
                
                # Show all candidates' scores
                st.subheader("Candidate Similarity Scores")
                for name, score in sorted(resume_scores.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{name}:** {score:.4f}")
            else:
                st.error("No valid text found after preprocessing.")
        else:
            st.error("No valid resume text extracted.")
    else:
        st.error("No text could be extracted from the job description.")
else:
    st.warning("Please upload resumes and a job description to proceed.")

