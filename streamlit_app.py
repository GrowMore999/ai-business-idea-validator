# streamlit_app.py
"""
AI Business Idea Validator - Streamlit UI (Plotly removed)
Use this file if plotly is not installed. Uses Streamlit's native st.bar_chart instead.
"""

import time
import io
import json
import urllib.parse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re  # used in preprocessing

# -----------------------------
# App config & CSS tokens
# -----------------------------
st.set_page_config(page_title="AI Business Idea Validator", layout="wide", initial_sidebar_state="collapsed")

CSS = """
:root{
  --color-primary: #0F62FE;
  --color-accent: #FF8A65;
  --color-neutral-900: #101418;
  --color-neutral-500: #6B7280;
  --bg-surface: #FFFFFF;
  --bg-soft: #F7FAFC;
  --radius: 12px;
  --space-4: 16px;
  --space-6: 24px;
  --shadow-soft: 0 6px 18px rgba(16,20,24,0.06);
  --shadow-card: 0 8px 30px rgba(16,20,24,0.08);
}

/* Layout */
.reportview-container .main .block-container{
  padding-top: 18px;
  padding-left: 20px;
  padding-right: 20px;
}
.header { display:flex; align-items:center; gap: 14px; margin-bottom: 8px; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:44px; height:44px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center; background:var(--color-primary); color:white; font-weight:700; box-shadow: var(--shadow-soft); }

/* Grid */
.app-shell { display:grid; grid-template-columns: 360px 1fr; gap: 28px; align-items:start; }
.input-card { background:var(--bg-surface); border-radius: var(--radius); padding: var(--space-6); box-shadow: var(--shadow-card); min-height: 420px; }
.small { font-size:13px; color:var(--color-neutral-500); }
.results { display:flex; flex-direction:column; gap: 18px; }
.result-card { background:var(--bg-surface); border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow-soft); }
.metric-big { display:flex; align-items:center; gap: 20px; }
.metric-number { font-size:48px; font-weight:700; color:var(--color-primary); }
.pill { padding:6px 12px; border-radius:999px; font-size:13px; font-weight:600; }
.chips { display:flex; gap:8px; flex-wrap:wrap; }

/* --- Hide Empty Chips --- */
.chip:empty,
.keyword:empty {
    display: none !important;
}

/* --- Highlight Chips with Content --- */
.chip {
    padding:6px 10px;
    background: #d6e4ff !important;
    color: #0f1a33 !important;
    border: 1px solid #A0B9FF;
    border-radius:999px;
    font-size:13px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.keyword {
    padding:6px 10px;
    background: #FFEFD6 !important;
    color: #4A2A00 !important;
    border: 1px solid #FFC78A;
    border-radius:999px;
    font-family:monospace;
    font-size:13px;
}

.btn { display:inline-flex; align-items:center; gap:8px; padding:8px 14px; border-radius:10px; cursor:pointer; border:none; }
.btn-primary { background:var(--color-primary); color:white; box-shadow: 0 6px 18px rgba(15,98,254,0.12); }
.btn-ghost { background:#2C3544; border:1px solid #3A4555; color:#E5E7EB; }
:focus { outline: 3px solid rgba(15,98,254,0.14); outline-offset: 3px; border-radius: 6px; }

@media (max-width: 980px) { .app-shell { grid-template-columns: 1fr; } }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# -----------------------------
# Training data and model
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_training_data() -> pd.DataFrame:
    data = [
        ("Subscription-based healthy meal prep for office workers", "Food & Beverage", "High"),
        ("B2B SaaS for automating invoice reconciliation for SMEs", "Software/SaaS", "High"),
        ("Telehealth platform for remote mental health consultations", "Healthcare", "High"),
        ("Fintech app for salary advances with employer integration", "Fintech", "High"),
        ("On-demand home cleaning app for urban families", "Home Services", "Medium"),
        ("Tutoring marketplace for school students", "Education", "Medium"),
        ("E-commerce store selling eco-friendly stationery", "E-commerce", "Medium"),
        ("Generic social media app for everyone", "Other", "Low"),
        ("Website that shows random quotes", "Other", "Low"),
        ("Simple blog about my daily life", "Other", "Low"),
    ]
    df = pd.DataFrame(data, columns=["text", "industry", "label"])
    df["combined"] = df["text"] + " [INDUSTRY] " + df["industry"]
    return df

@st.cache_resource(show_spinner=False)
def train_model() -> Tuple[Pipeline, Dict]:
    df = get_training_data()
    X = df["combined"]
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=400))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return pipeline, report

model, model_report = train_model()

# -----------------------------
# XAI helpers
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
GENERIC_WORDS = {"app", "website", "platform", "service", "online", "digital", "solution", "system", "business", "idea", "startup", "portal"}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_keywords(text: str, top_k: int = 7) -> List[str]:
    text = preprocess(text)
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
        "Keyword richness": int(round(keyword_richness * 100)),
        "Industry demand": int(round(industry_demand * 100)),
        "Novelty": int(round(novelty * 100)),
        "Simplicity": int(round(simplicity * 100)),
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
    else:
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
# Sidebar + header
# -----------------------------
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:12px'><div class='logo'>A</div><div><strong>AI Business Idea Validator</strong><div class='small'>Demo • TF-IDF + Logistic Regression</div></div></div>", unsafe_allow_html=True)
    st.caption("Demo model uses small internal dataset for stability.")
    st.write("---")
    if model_report:
        with st.expander("Model details (for viva)"):
            st.write("Model: Logistic Regression on TF-IDF features.")
            st.write(f"Data Size: {len(get_training_data())} rows (internal demo set).")
            st.json(model_report, expanded=False)
    st.write("---")
    if st.button("Download design_tokens.json"):
        tokens = {
            "colors": {"primary": "#0F62FE", "accent": "#FF8A65", "neutral-900": "#101418"},
            "spacing": {"4": 16, "6": 24}, "radius": 12
        }
        st.download_button("Download JSON", data=json.dumps(tokens, indent=2), file_name="design_tokens.json", mime="application/json")

st.markdown("""
<div class="header">
  <div class="brand">
    <div class="logo" aria-hidden="true">AB</div>
    <div>
      <div style="font-weight:700">AI Business Idea Validator</div>
      <div style="font-size:13px;color:var(--color-neutral-500)">Quick, credible feedback for founders</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main layout
# -----------------------------
st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

st.markdown("<div class='input-card' role='region' aria-labelledby='input-title'>", unsafe_allow_html=True)
st.markdown("<h3 id='input-title'>1️⃣ Input your idea</h3>", unsafe_allow_html=True)

title = st.text_input("Business Idea Title", placeholder="e.g., Subscription-based healthy tiffin service for office workers", key="title_input", help="Short, specific title describing your idea in one line.")
desc = st.text_area("Describe your idea", placeholder="What problem do you solve? Who is the customer? How does it work?", height=160, key="desc_input", help="Include customer, problem, and how you solve it. Aim for 40-120 words.")
industry = st.selectbox("Industry", ["Food & Beverage", "E-commerce", "Home Services", "Education","Healthcare", "Fintech", "Software/SaaS", "Other"], index=0)
goal = st.radio("What is your current goal?", ["Market validation", "Competition analysis", "Funding readiness"], horizontal=False)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1])
with col_a:
    analyze_btn = st.button("🔍 Analyze Idea", key="analyze", help="Run the model and show explanation metrics", use_container_width=True)
with col_b:
    reset_btn = st.button("Reset inputs", key="reset_btn", help="Clear inputs")
    if reset_btn:
        st.session_state["title_input"] = ""
        st.session_state["desc_input"] = ""
        st.rerun()

st.markdown("<div class='small' style='margin-top:12px'>Tip: Use concrete nouns & numbers. E.g., '50 corporate clients in Mumbai'.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close input-card

st.markdown("<div class='results' role='region' aria-live='polite'>", unsafe_allow_html=True)
st.markdown("<div class='result-card'><div style='display:flex;justify-content:space-between;align-items:center'><div><h3 style='margin:0'>2️⃣ AI Evaluation & Insights</h3><div class='small'>Actionable guidance with model + rule-based explanations</div></div></div></div>", unsafe_allow_html=True)

results_placeholder = st.empty()

if analyze_btn:
    if not title.strip() or not desc.strip():
        st.warning("Please fill in both the title and description.")
    else:
        with results_placeholder.container():
            st.markdown("<div class='result-card'><div style='height:22px;width:70%;background:#eef3ff;border-radius:8px;margin-bottom:14px'></div><div style='height:14px;width:40%;background:#f3f5f7;border-radius:6px;margin-bottom:22px'></div><div style='height:160px;background:#fbfcfe;border-radius:10px'></div></div>", unsafe_allow_html=True)
        time.sleep(0.45)

        full_text = f"{title} {desc}"
        combined = f"{title} {desc} [INDUSTRY] {industry}"
        pred_label = model.predict([combined])[0]
        proba = model.predict_proba([combined])[0]
        classes = model.classes_
        label_to_weight = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
        weighted_score = sum(label_to_weight[c] * p for c, p in zip(classes, proba))
        overall_score = int(round(weighted_score * 100))

        feature_details, keywords = compute_explanatory_features(title, desc, industry)
        models = suggest_business_models(full_text)
        risks = identify_risks(industry, full_text)
        steps = suggest_next_steps(pred_label, goal)

        prob_df = pd.DataFrame({"Category": classes, "Probability": np.round(proba * 100, 1)})
        feat_items = {k: v for k, v in feature_details.items() if k != "Length (words)"}
        feat_df = pd.DataFrame({"Feature": list(feat_items.keys()), "Score": list(feat_items.values())})

        with results_placeholder.container():
            st.markdown("<div class='result-card' role='region' aria-label='Top metrics'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1.6, 1])
            with col1:
                score_holder = st.empty()
                score_holder.markdown(f"<div class='metric-big'><div><div class='metric-number'>{overall_score}</div><div class='small'>Overall Feasibility Score</div></div><div style='display:flex;flex-direction:column;align-items:flex-start;gap:8px'><div class='pill' style='background:#EBF4FF;color:var(--color-primary)'>Predicted: <strong style='margin-left:8px'>{pred_label}</strong></div><div class='small'>Model confidence: {int(round(max(proba)*100))}%</div></div></div>", unsafe_allow_html=True)
                for v in range(0, overall_score + 1, max(1, overall_score // 20 or 1)):
                    score_holder.markdown(f"<div class='metric-big'><div><div class='metric-number'>{v}</div><div class='small'>Overall Feasibility Score</div></div><div style='display:flex;flex-direction:column;align-items:flex-start;gap:8px'><div class='pill' style='background:#EBF4FF;color:var(--color-primary)'>Predicted: <strong style='margin-left:8px'>{pred_label}</strong></div><div class='small'>Model confidence: {int(round(max(proba)*100))}%</div></div></div>", unsafe_allow_html=True)
                    time.sleep(0.01)
            with col2:
                st.markdown("<div style='display:flex;flex-direction:column;gap:8px;align-items:flex-end'><button class='btn btn-ghost' onclick='window.print()'>Print report</button></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Charts using Streamlit native st.bar_chart
            st.markdown("<div style='display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:12px'>", unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><strong>Class Probabilities</strong><div class='small'>Model output</div></div>", unsafe_allow_html=True)
                prob_plot_df = prob_df.set_index("Category")
                st.bar_chart(prob_plot_df)
                csv_buf = prob_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download probabilities CSV", data=csv_buf, file_name="probabilities.csv", mime="text/csv")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><strong>Feature Breakdown</strong><div class='small'>Explainable features</div></div>", unsafe_allow_html=True)
                feat_plot_df = feat_df.set_index("Feature")
                st.bar_chart(feat_plot_df)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px'>", unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<strong>Extracted keywords</strong>")
                if keywords:
                    chip_html = "<div class='chips' style='margin-top:8px'>"
                    for k in keywords:
                        if k.strip():  # only print if not empty
                            chip_html += f"<div class='keyword'>{k}</div>"
                    chip_html += "</div>"
                    st.markdown(chip_html, unsafe_allow_html=True)
                else:
                    st.info("No strong keywords extracted. Try using more specific, concrete words.")
                st.markdown("<hr style='margin-top:12px;margin-bottom:12px'/>", unsafe_allow_html=True)
                st.markdown("<strong>Suggested business model(s)</strong>")
                model_chips = "<div style='margin-top:8px' class='chips'>"
                for m in models:
                    if m.strip():  # only print if not empty
                        model_chips += f"<div class='chip'>{m}</div>"
                model_chips += "</div>"
                st.markdown(model_chips, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<strong>Key risks</strong>")
                for r in risks:
                    st.markdown(f"- {r}")
                st.markdown("<hr style='margin-top:12px;margin-bottom:12px'/>", unsafe_allow_html=True)
                st.markdown("<strong>Recommended next steps</strong>")
                for i, s in enumerate(steps, 1):
                    st.markdown(f"{i}. {s}")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='result-card' style='display:flex;flex-direction:column;gap:8px'>", unsafe_allow_html=True)
            st.markdown("<strong>Quick research links</strong>", unsafe_allow_html=True)
            comp_query = f"{industry} {title} competitors"
            market_query = f"{industry} market size report"
            links_html = f"""
            <div style='display:flex;gap:10px;margin-top:8px;flex-wrap:wrap'>
              <a class='btn btn-ghost' href="{google_search_link(comp_query)}" target="_blank" rel="noopener">Search competitors</a>
              <a class='btn btn-ghost' href="{google_search_link(market_query)}" target="_blank" rel="noopener">Market size / trends</a>
              <a class='btn btn-ghost' href="{google_search_link(title + ' startup idea')}" target="_blank" rel="noopener">Similar startup ideas</a>
            </div>
            """
            st.markdown(links_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Technical explanation (for report / viva)"):
                st.markdown("""
                **Model architecture**
                - TF-IDF vectorization of title + description + industry.
                - Logistic Regression classifier outputs class probabilities.
                - Rule-based XAI features for interpretability.

                **Handoff notes**
                - Design tokens available via sidebar download.
                - Charts use Streamlit native bar_chart (no external libs required).
                - Accessibility: form inputs include descriptive labels and help text.
                """)

st.markdown("</div>", unsafe_allow_html=True)  # close results
st.markdown("</div>", unsafe_allow_html=True)  # close app-shell

st.markdown("""<div style="margin-top:18px;color:var(--color-neutral-500);font-size:13px">Designed for founders • Demo model uses a small internal dataset • Replace demo data with your GitHub dataset for production.</div>""", unsafe_allow_html=True)
