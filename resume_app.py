import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Avoid NoneType issues
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Function to calculate ATS score
def calculate_ats_score(resume_text, job_description):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_description.lower().split())

    matched_words = resume_words.intersection(job_words)
    ats_score = (len(matched_words) / len(job_words)) * 100 if job_words else 0  # Handle empty job description
    
    return round(ats_score, 2)

# Function to suggest resume improvements
def suggest_resume_improvements(resume_text):
    suggestions = []
    
    keywords = ["teamwork", "leadership", "projects", "certification", "communication"]
    for word in keywords:
        if word not in resume_text.lower():
            suggestions.append(f"🔹 Add {word}-related skills or achievements.")

    return "\n".join(suggestions) if suggestions else "✅ Your resume looks well-structured!"

# Streamlit UI
st.set_page_config(page_title="ResumeRanker - Resume Screening App", layout="wide")

st.title("📄 ResumeRanker: Resume Screening App 🎯")

# **Instruction Box**
st.markdown("""
    🟢 **How to Use:**  
    1️⃣ **Enter the Job Description** in the left panel.  
    2️⃣ **Upload Resume PDFs** below.  
    3️⃣ **Press `Ctrl + Enter`** to check rankings & ATS scores.  
""", unsafe_allow_html=True)

# Job Description Input
st.sidebar.header("📝 Enter Job Description")
job_description = st.sidebar.text_area("Enter the job description")

# Resume Upload
st.sidebar.header("📂 Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload multiple PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("📊 Resume Ranking & ATS Analysis")
    
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    
    # Rank resumes
    scores = rank_resumes(job_description, resumes)
    
    # Calculate ATS scores
    ats_scores = [calculate_ats_score(resume, job_description) for resume in resumes]
    
    # Generate resume improvement suggestions
    suggestions_list = [suggest_resume_improvements(resume) for resume in resumes]

    # Assign labels (Resume 1, Resume 2...) instead of file names
    resume_labels = [f"Resume {i+1}" for i in range(len(uploaded_files))]

    # Display scores in a structured table
    results = pd.DataFrame({
        "Resume": resume_labels,  # Use labels instead of file names
        "AI Match Score (0-1)": scores,
        "ATS Score (%)": ats_scores,
        "Suggestions": suggestions_list
    })
    results = results.sort_values(by="AI Match Score (0-1)", ascending=False)

    # Display results as a well-structured table
    st.write("### 📌 Resume Rankings & ATS Scores")
    st.dataframe(results.style.set_properties(**{'text-align': 'left'}))

    # **Fix X-axis label issue** by setting rotation & alignment properly
    st.write("### 📊 ATS Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=resume_labels, y=results["ATS Score (%)"], palette="coolwarm", ax=ax)

    # **Improve X-axis readability**
    ax.set_xticklabels(resume_labels, rotation=25, ha="right")  # Rotate properly
    plt.ylabel("ATS Score (%)")
    plt.xlabel("Resume")
    plt.title("ATS Score Comparison for Uploaded Resumes")

    st.pyplot(fig)
