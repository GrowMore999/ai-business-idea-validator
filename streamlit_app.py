import streamlit as st
import re

st.title("🤖 AI Business Idea Validator")

INDUSTRY_DEMAND = {
    "Food & Beverage": 0.8,
    "E-commerce": 0.7,
    "Home Services": 0.6,
    "Education": 0.7,
    "Healthcare": 0.75
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extract_keywords(text):
    words = text.split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_w = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w,_ in sorted_w[:5]]

def compute_scores(title, desc, industry):
    text = preprocess(title + " " + desc)
    keywords = extract_keywords(text)
    keyword_richness = min(1.0, len(set(keywords))/5.0)
    industry_score = INDUSTRY_DEMAND.get(industry, 0.5)
    score = int((keyword_richness*0.4 + industry_score*0.6) * 100)
    return score, keywords

title = st.text_input("Business Idea Title")
desc = st.text_area("Describe your business idea")
industry = st.selectbox("Select Industry", list(INDUSTRY_DEMAND.keys()))

if st.button("Analyze Idea"):
    score, keywords = compute_scores(title, desc, industry)
    st.subheader(f"✅ Score: {score}/100")
    st.write("📌 Keywords:", keywords)
    st.write("📊 Market Potential:",
             "High" if score >= 70 else "Medium" if score >= 40 else "Low")
    st.success("🔥 Recommended next steps:\n1. Create landing page\n2. Collect 100 signups\n3. Test with small pilot")
