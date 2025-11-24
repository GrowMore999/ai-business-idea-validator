import re
from typing import List, Tuple, Dict

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
st.caption("ML + NLP based scoring for startup / business ideas")


# -----------------------------
# 1. Training data (toy dataset)
#    You can expand this later or load from CSV.
# -----------------------------
def get_training_data() -> pd.DataFrame:
    data = [
        # High potential examples
        ("Subscription-based healthy meal prep for office workers", "Food & Beverage", "High"),
        ("B2B SaaS for automating invoice reconciliation for SMEs", "Software/SaaS", "High"),
        ("Telehealth platform for remote mental health consultations", "Healthcare", "High"),
        ("Fintech app for salary advances with employer integration", "Fintech", "High"),
        ("Marketplace connecting local farmers to urban restaurants", "Food & Beverage", "High"),

        # Medium potential examples
        ("On-demand home cleaning app for urban families", "Home Services", "Medium"),
        ("Tutoring marketplace for school students", "Education", "Medium"),
        ("E-commerce store selling eco-friendly stationery", "E-commerce", "Medium"),
        ("Digital marketing service for small local shops", "Software/SaaS", "Medium"),
        ("Online language course platform with live classes", "Education", "Medium"),

        # Low potential examples
        ("Generic social media app for everyone", "Other", "Low"),
        ("Website that shows random quotes", "Other", "Low"),
        ("Another food delivery app with no new features", "Food & Beverage", "Low"),
        ("Simple blog about my daily life", "Other", "Low"),
        ("Basic online shop without niche focus", "E-commerce", "Low"),
    ]
    df = pd.DataFrame(data, columns=["text", "industry", "label"])
    # Combine text + industry to give model more signal
    df["combined"] = df["text"] + " [INDUSTRY] " + df["industry"]
    return df


# -----------------------------
# 2. Model training (TF-IDF + Logistic Regression)
# -----------------------------
@st.cache_resource(show_spinner=False)
def train_model() -> Tuple[Pipeline, Dict[str, Dict]]:
    df = get_training_data()
    X = df["combined"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on validation set (for report)
    y_pred = pipeline.predict(X_val)
    report_dict = classification_report(
        y_val, y_pred, output_dict=True, zero_division=0
    )

    return pipeline, report_dict


model, model_report = train_model()


# -----------------------------
# 3. Auxiliary "AI explanation" features
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
        "Novelty (less generic)": round(novelty * 100),
        "Simplicity (clarity)": round(simplicity * 100),
        "Length (word count)": word_count,
    }

    return feature_details, keywords


def suggest_business_models(text: str) -> List[str]:
    t = text.lower()
    models = []
    if any(w in t for w in ["subscription", "monthly", "saas"]):
        models.append("Subscription model")
    if any(w in t for w in ["marketplace", "buyers", "sellers", "listing", "connect"]):
        models.append("Marketplace model")
    if any(w in t for w in ["delivery", "on-demand", "on demand", "logistics"]):
        models.append("On-demand / delivery model")
    if any(w in t for w in ["course", "learn", "training", "tutorial", "academy"]):
        models.append("Online course / cohort model")
    if not models:
        models.append("Direct-to-consumer (D2C)")
    return models


def identify_risks(industry: str, text: str) -> List[str]:
    t = text.lower()
    risks = []
    if industry == "Food & Beverage":
        risks.append("Perishable inventory and food safety requirements.")
    if industry in ["E-commerce", "Software/SaaS", "Fintech"]:
        risks.append("High competition – strong differentiation and marketing needed.")
    if "delivery" in t or "on-demand" in t:
        risks.append("Operational complexity in logistics and last-mile delivery.")
    if "subscription" in t:
        risks.append("Churn risk – customers may cancel if value is not sustained.")
    if len(risks) == 0:
        risks.append("Need to validate real customer demand and willingness to pay.")
    return risks


def suggest_next_steps(pred_label: str) -> List[str]:
    if pred_label == "High":
        return [
            "Build a simple landing page and collect at least 100 signups.",
            "Interview 5–10 target customers to refine feature set.",
            "Create a small MVP and test one pricing option."
        ]
    elif pred_label == "Medium":
        return [
            "Narrow down to a specific niche or customer segment.",
            "Validate the core problem via customer interviews.",
            "Test the idea using a no-code prototype or mockups."
        ]
    else:
        return [
            "Clarify the exact problem and target customer persona.",
            "Study at least 3–5 competitors or similar solutions.",
            "Refine the value proposition to be more specific and unique."
        ]


# -----------------------------
# 4. UI Layout
# -----------------------------

# Sidebar – general info
with st.sidebar:
    st.header("About this tool")
    st.write(
        "This is a **machine learning–based** evaluator for business ideas. "
        "It uses a TF-IDF + Logistic Regression model trained on sample startup "
        "ideas (High / Medium / Low potential)."
    )
    st.write("Technologies:")
    st.code("Python · Streamlit · scikit-learn · TF-IDF · Logistic Regression", language="bash")

    with st.expander("Model details (for viva)"):
        st.write("Model: Logistic Regression on TF-IDF features.")
        st.write("Labels: High, Medium, Low potential.")
        st.json(model_report, expanded=False)


# Main inputs & outputs
input_col, result_col = st.columns([1.1, 1.3])

with input_col:
    st.subheader("Enter your business idea")

    title = st.text_input(
        "Business Idea Title",
        placeholder="e.g., Subscription-based healthy tiffin service for office workers"
    )
    desc = st.text_area(
        "Describe your idea",
        placeholder="What problem do you solve? Who is the customer? How does it work?",
        height=200
    )
    industry = st.selectbox(
        "Industry",
        ["Food & Beverage", "E-commerce", "Home Services", "Education",
         "Healthcare", "Fintech", "Software/SaaS", "Other"]
    )

    analyze_btn = st.button("Analyze Idea")

with result_col:
    st.subheader("AI Evaluation")

    if analyze_btn:
        if not title.strip() or not desc.strip():
            st.warning("Please fill in both the title and the description.")
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
            steps = suggest_next_steps(pred_label)

            # Top-level metrics
            top1_col, top2_col = st.columns(2)
            with top1_col:
                st.metric("Overall Feasibility Score", f"{overall_score}/100")
            with top2_col:
                st.metric("Predicted Potential Category", pred_label)

            # Probability breakdown table
            st.markdown("#### Class probabilities (ML model output)")
            prob_df = pd.DataFrame({
                "Category": classes,
                "Probability": np.round(proba * 100, 1)
            }).sort_values("Probability", ascending=False)
            st.table(prob_df)

            # Feature breakdown
            st.markdown("#### Feature breakdown (for explanation)")
            fb1, fb2 = st.columns(2)
            with fb1:
                st.write(f"- Keyword richness: **{feature_details['Keyword richness']} / 100**")
                st.write(f"- Industry demand: **{feature_details['Industry demand']} / 100**")
            with fb2:
                st.write(f"- Novelty (less generic): **{feature_details['Novelty (less generic)']} / 100**")
                st.write(f"- Simplicity / clarity: **{feature_details['Simplicity (clarity)']} / 100**")
                st.write(f"- Length: **{feature_details['Length (word count)']} words**")

            # Keywords
            st.markdown("#### Extracted keywords (NLP)")
            if keywords:
                st.write(", ".join(f"`{k}`" for k in keywords))
            else:
                st.write("_No strong keywords extracted. Try adding more specific details._")

            # Suggested business models
            st.markdown("#### Suggested business model(s)")
            for m in models:
                st.write(f"- {m}")

            # Risks
            st.markdown("#### Key risks to consider")
            for r in risks:
                st.write(f"- {r}")

            # Recommended next steps
            st.markdown("#### Recommended next steps")
            for s in steps:
                st.write(f"- {s}")

            # Technical explanation section
            with st.expander("Technical explanation (for report / viva)"):
                st.markdown(
                    """
                    **1. ML model architecture**

                    - Text is converted into numerical features using **TF-IDF vectorization**
                      (unigrams + bigrams).
                    - A **Logistic Regression** classifier is trained on labeled examples
                      of ideas (High / Medium / Low potential).
                    - For a new idea, the model outputs:
                      - A predicted label (High / Medium / Low)
                      - Probabilities for each label.

                    **2. Additional feature engineering (NLP-style):**

                    - The system also computes explainable features:
                      - Keyword richness (distinct important words).
                      - Industry demand (predefined weights per industry).
                      - Novelty (penalizes overly generic buzzwords).
                      - Simplicity (based on description length).

                    **3. Final score calculation**

                    - The class probabilities are combined using weights:
                      - Low = 0.3, Medium = 0.6, High = 1.0
                    - The weighted sum is scaled to a 0–100 score.

                    This combination of a trained ML model + rule-based explanations
                    makes the system both **data-driven** and **interpretable**, which
                    is ideal for a software engineering project.
                    """
                )
    else:
        st.info("Fill in your idea on the left and click **Analyze Idea** to see the model output here.")

