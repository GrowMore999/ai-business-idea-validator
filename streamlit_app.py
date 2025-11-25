import re
from typing import List, Tuple, Dict
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="AI Business Idea Validator",
    layout="wide"
)

st.title("AI Business Idea Validator")
st.caption("ML + NLP based scoring and analysis for startup / business ideas")

# -----------------------------
# 1. Training data (Small Internal Dataset - GUARANTEED TO RUN)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_training_data() -> pd.DataFrame:
    """Uses a small, hardcoded dataset for stable demonstration."""
    data = [
        # High potential examples (Strong Keywords, High Demand Industry)
        ("Subscription-based healthy meal prep for office workers", "Food & Beverage", "High"),
        ("B2B SaaS for automating invoice reconciliation for SMEs", "Software/SaaS", "High"),
        ("Telehealth platform for remote mental health consultations", "Healthcare", "High"),
        ("Fintech app for salary advances with employer integration", "Fintech", "High"),

        # Medium potential examples
        ("On-demand home cleaning app for urban families", "Home Services", "Medium"),
        ("Tutoring marketplace for school students", "Education", "Medium"),
        ("E-commerce store selling eco-friendly stationery", "E-commerce", "Medium"),

        # Low potential examples (Generic, Low Demand Industry)
        ("Generic social media app for everyone", "Other", "Low"),
        ("Website that shows random quotes", "Other", "Low"),
        ("Simple blog about my daily life", "Other", "Low"),
    ]
    df = pd.DataFrame(data, columns=["text", "industry", "label"])
    df["combined"] = df["text"] + " [INDUSTRY] " + df["industry"]
    st.sidebar.success(f"Demo Model trained on {len(df)} internal data rows.")
    return df

# -----------------------------
# 2. Model training (TF-IDF + Logistic Regression)
# -----------------------------
@st.cache_resource(show_spinner=False)
def train_model() -> Tuple[Pipeline, Dict[str, Dict]]:
    with st.spinner("Training ML Model on internal data..."):
        df = get_training_data()
        
        # We ensure the DataFrame is not empty (which it won't be with hardcoded data)
        X = df["combined"]
        y = df["label"]

        # Smaller test_size as the total dataset is small
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=200)) 
        ])
        
        pipeline.fit(X_train, y_train) 

        # Evaluate on validation set
        y_pred = pipeline.predict(X_val)
        report_dict = classification_report(
            y_val, y_pred, output_dict=True, zero_division=0
        )

        return pipeline, report_dict


# Simple model assignment (no try/except needed since data is internal)
model, model_report = train_model()

# -----------------------------
# 3. Auxiliary "AI explanation" features (Rules-based AI for XAI)
# -----------------------------
INDUSTRY_DEMAND = {
    "Food & Beverage": 0.8,
    "E-commerce": 0.75,
    "Home Services": 0.6,
    "Education": 0.7,
    "Healthcare": 0.8,
    "Fintech": 0.8,
    "Software/SaaS": 0.78,
    "Other": 0.5
}

GENERIC_WORDS = {
    "app", "website", "platform", "service", "online", "digital",
    "solution", "system", "business", "idea", "startup", "portal"
}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_keywords(text: str, top_k: int = 7) -> List[str]:
    words = text.split()
    freq = {}
    for w in words:
        if len(w) <= 3:
            continue
        freq[w] = freq.get(w, 0) + 1
    sorted_w = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_w[:top_k]]

def compute_explanatory_features(title: str, desc: str, industry: str) -> Tuple[Dict, List[str]]:
    full_text = preprocess(title + " " + desc)
    words = full_text.split()

    keywords = extract_keywords(full_text)

    keyword_richness = min(1.0, len(set(keywords)) / 7.0)
    industry_demand = INDUSTRY_DEMAND.get(industry, INDUSTRY_DEMAND["Other"])

    generic_count = sum(1 for k in keywords if k in GENERIC_WORDS)
    if len(keywords) == 0:
        novelty = 0.4
    else:
        novelty = max(0.2, 1.0 - generic_count / max(1, len(keywords)))

    word_count = len(words)
    if word_count <= 40:
        simplicity = 0.9
    elif word_count <= 120:
        simplicity = 0.7
    else:
        simplicity = 0.5

    feature_details = {
        "Keyword richness": round(keyword_richness * 100),
        "Industry demand": round(industry_demand * 100),
        "Novelty": round(novelty * 100),
        "Simplicity": round(simplicity * 100),
        "Length (words)": word_count,
    }

    return feature_details, keywords

def suggest_business_models(text: str) -> List[str]:
    t = text.lower()
    models = []
    if any(w in t for w in ["subscription", "monthly", "saas"]):
        models.append("Subscription")
    if any(w in t for w in ["marketplace", "buyers", "sellers", "listing", "connect"]):
        models.append("Marketplace")
    if any(w in t for w in ["delivery", "on-demand", "on demand", "logistics"]):
        models.append("On-demand / delivery")
    if any(w in t for w in ["course", "learn", "training", "tutorial", "academy"]):
        models.append("Online course / cohort")
    if not models:
        models.append("Direct-to-consumer (D2C)")
    return models

def identify_risks(industry: str, text: str) -> List[str]:
    t = text.lower()
    risks = []
    if industry == "Food & Beverage":
        risks.append("Perishable inventory and food safety requirements.")
    if industry in ["E-commerce", "Software/SaaS", "Fintech"]:
        risks.append("High competition – strong differentiation needed.")
    if "delivery" in t or "on-demand" in t:
        risks.append("Operational complexity in logistics and last-mile delivery.")
    if "subscription" in t:
        risks.append("Churn risk – customers may cancel if value drops.")
    if len(risks) == 0:
        risks.append("Need to validate real customer demand and willingness to pay.")
    return risks

def suggest_next_steps(pred_label: str, goal: str) -> List[str]:
    if goal == "Market validation":
        base = [
            "Interview at least 5–10 target customers.",
            "Create a simple landing page and measure signups.",
            "Test whether people understand the value in 10 seconds."
        ]
    elif goal == "Competition analysis":
        base = [
            "Search for top 5 competitors and list their strengths/weaknesses.",
            "Identify at least 2–3 clear differentiators for your idea.",
            "Check pricing and positioning of similar tools."
        ]
    else:  # Funding readiness
        base = [
            "Prepare a 1-page problem/solution/market summary.",
            "Estimate basic unit economics (how you make money).",
            "Collect early traction metrics (signups, waitlist, pilots)."
        ]

    if pred_label == "High":
        base.insert(0, "Double down: your idea seems promising, focus on execution.")
    elif pred_label == "Medium":
        base.insert(0, "Refine positioning: idea has potential but needs sharper focus.")
    else:
        base.insert(0, "Rework the concept: current version looks weak; sharpen the problem and niche.")

    return base

def google_search_link(query: str) -> str:
    return "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)

# -----------------------------
# 4. UI Layout
# -----------------------------
# Sidebar – general info
with st.sidebar:
    st.header("About this tool")
    st.write(
        "This tool uses **TF-IDF + Logistic Regression** to classify ideas into "
        "*High / Medium / Low* potential. The demo uses a small internal dataset for fast loading."
    )
    st.write("Tech stack:")
    st.code("Python · Streamlit · scikit-learn · TF-IDF · LogisticRegression")

    if model_report:
        with st.expander("Model details (for viva)"):
            st.write("Model: Logistic Regression on TF-IDF features.")
            st.write(f"Data Size: {len(get_training_data())} Rows (Internal Demo Set).")
            st.json(model_report, expanded=False)

# The model is guaranteed to exist now, so we run the main app.
# Main inputs & outputs
input_col, result_col = st.columns([1.1, 1.3])

with input_col:
    st.subheader("1️⃣ Input your idea")

    title = st.text_input(
        "Business Idea Title",
        placeholder="e.g., Subscription-based healthy tiffin service for office workers"
    )
    desc = st.text_area(
        "Describe your idea",
        placeholder="What problem do you solve? Who is the customer? How does it work?",
        height=180
    )
    industry = st.selectbox(
        "Industry",
        ["Food & Beverage", "E-commerce", "Home Services", "Education",
         "Healthcare", "Fintech", "Software/SaaS", "Other"]
    )
    goal = st.radio(
        "What is your current goal?",
        ["Market validation", "Competition analysis", "Funding readiness"],
        horizontal=False
    )

    analyze_btn = st.button("🔍 Analyze Idea")

with result_col:
    st.subheader("2️⃣ AI Evaluation & Insights")

    if analyze_btn:
        if not title.strip() or not desc.strip():
            st.warning("Please fill in both the title and description.")
        else:
            full_text = f"{title} {desc}"
            combined = f"{title} {desc} [INDUSTRY] {industry}"

            # 1) ML model prediction
            pred_label = model.predict([combined])[0]
            proba = model.predict_proba([combined])[0]
            classes = model.classes_

            # Overall score from probabilities
            label_to_weight = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
            weighted_score = sum(
                label_to_weight[c] * p for c, p in zip(classes, proba)
            )
            overall_score = int(weighted_score * 100)

            # 2) Explanation features
            feature_details, keywords = compute_explanatory_features(title, desc, industry)
            models = suggest_business_models(full_text)
            risks = identify_risks(industry, full_text)
            steps = suggest_next_steps(pred_label, goal)

            # --- Core Metrics ---
            top1, top2 = st.columns(2)
            with top1:
                st.metric("Overall Feasibility Score", f"{overall_score}/100")
            with top2:
                st.metric("Predicted Category", pred_label)
            
            # Feature breakdown (Text only)
            st.markdown("#### Feature Breakdown (for explanation)")
            fb1, fb2 = st.columns(2)
            with fb1:
                st.write(f"- Keyword richness: **{feature_details['Keyword richness']} / 100**")
                st.write(f"- Industry demand: **{feature_details['Industry demand']} / 100**")
            with fb2:
                st.write(f"- Novelty (less generic): **{feature_details['Novelty']} / 100**")
                st.write(f"- Simplicity / clarity: **{feature_details['Simplicity']} / 100**")
                st.write(f"- Length: **{feature_details['Length (words)']} words**")
            
            # Keywords
            st.markdown("#### Extracted keywords")
            if keywords:
                st.write(", ".join(f"`{k}`" for k in keywords))
            else:
                st.write("_No strong keywords extracted. Try using more specific, concrete words._")
            
            # Suggested business models
            st.markdown("#### Suggested business model(s)")
            st.write(", ".join(models))
            
            # Risks
            st.markdown("#### Key risks")
            for r in risks:
                st.write(f"- {r}")
            
            # Recommended next steps
            st.markdown("#### Recommended next steps")
            for s in steps:
                st.write(f"- {s}")
            
            # --- Dedicated Visualization Link Section ---
            st.markdown("---")
            st.markdown("### 📊 Dedicated Visual Analysis Dashboard")
            st.markdown(
                "Click the expander below to see the **ML Probability Chart** and the **Feature Breakdown Chart**."
            )
            
            # Charts are now hidden inside this expander, simulating a separate link/page
            with st.expander("📈 Show Visual Dashboard"):
                st.subheader("Visual Analysis: Model & Features")
                
                # 1. Probability chart
                st.markdown("##### Class Probabilities (ML Output)")
                prob_df = pd.DataFrame({
                    "Category": classes,
                    "Probability": np.round(proba * 100, 1)
                }).sort_values("Probability", ascending=True)
                st.bar_chart(
                    prob_df.set_index("Category"),
                    height=200
                )

                # 2. Feature breakdown chart
                st.markdown("##### Feature Breakdown (Explanation Scores)")
                feat_df = pd.DataFrame(
                    {
                        "Feature": [k for k in feature_details.keys() if k != "Length (words)"],
                        "Score": [feature_details[k] for k in feature_details.keys() if k != "Length (words)"]
                    }
                )
                st.bar_chart(
                    feat_df.set_index("Feature"),
                    height=220
                )
                
            # Quick Google research links
            st.markdown("---")
            st.markdown("#### 🔗 Quick Google research links")
            comp_query = f"{industry} {title} competitors"
            market_query = f"{industry} market size report"
            st.markdown(f"- [Search competitors]({google_search_link(comp_query)})")
            st.markdown(f"- [Search market size / trends]({google_search_link(market_query)})")
            st.markdown(
                f"- [Search similar startup ideas]({google_search_link(title + ' startup idea')})"
            )

            # Technical explanation
            with st.expander("Technical explanation (for report / viva)"):
                st.markdown(
                    """
                    **Model Architecture**

                    - Text is converted using **TF-IDF vectorization**.
                    - **Logistic Regression** classifier is trained on the demo data to output **class probabilities**.

                    **Explainable AI (XAI)**
                    - Hand-crafted NLP features (Keyword richness, Novelty, etc.) are computed alongside the ML model to provide a clear, rule-based explanation for the score, enhancing **model interpretability**.
                    """
                )
    else:
        st.info("Enter your idea on the left and click **Analyze Idea** to see the model output here.")
