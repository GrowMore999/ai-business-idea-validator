# streamlit_app.py
"""
AI Business Idea Validator - Modern Streamlit UI
Single-file app: paste/replace your existing streamlit_app.py with this file.

Features:
- Preserves ML model (TF-IDF + LogisticRegression) trained on internal demo data
- Modern left input panel / right results panel layout
- Plotly charts for probabilities & feature breakdown
- Keyword chips, suggested business models, risks, actionable next steps
- Accessibility-friendly form labels and keyboard focus styles
- CSS tokens (Tailwind-friendly variable names) included at top
- Lightweight animations: count-up score, skeleton loading for charts
- Export buttons to copy content / download CSV of probabilities
"""

import time
import io
import json
import urllib.parse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# App config & CSS tokens
# -----------------------------
st.set_page_config(page_title="AI Business Idea Validator", layout="wide", initial_sidebar_state="collapsed")

# Design tokens (Tailwind-friendly variable names)
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
  --glass: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(255,255,255,0.6));
}

/* Reset some streamlit defaults for a cleaner look */
.reportview-container .main .block-container{
  padding-top: 18px;
  padding-left: 20px;
  padding-right: 20px;
}

/* Header */
.header {
  display:flex;
  align-items:center;
  gap: 14px;
  margin-bottom: 8px;
}
.brand {
  display:flex;
  align-items:center;
  gap:12px;
}
.logo {
  width:44px;
  height:44px;
  border-radius:10px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  background:var(--color-primary);
  color:white;
  font-weight:700;
  box-shadow: var(--shadow-soft);
}

/* Layout columns */
.app-shell {
  display:grid;
  grid-template-columns: 360px 1fr;
  gap: 28px;
  align-items:start;
}

/* Left input card */
.input-card {
  background:var(--bg-surface);
  border-radius: var(--radius);
  padding: var(--space-6);
  box-shadow: var(--shadow-card);
  min-height: 420px;
}
.small {
  font-size:13px;
  color:var(--color-neutral-500);
}

/* Right results card */
.results {
  display:flex;
  flex-direction:column;
  gap: 18px;
}
.result-card {
  background:var(--bg-surface);
  border-radius: var(--radius);
  padding: 18px;
  box-shadow: var(--shadow-soft);
}

/* Metric big */
.metric-big {
  display:flex;
  align-items:center;
  gap: 20px;
}
.metric-number {
  font-size:48px;
  font-weight:700;
  color:var(--color-primary);
}
.pill {
  padding:6px 12px;
  border-radius:999px;
  font-size:13px;
  font-weight:600;
}

/* Chips / pills */
.chips {
  display:flex;
  gap:8px;
  flex-wrap:wrap;
}
.chip {
  padding:6px 10px;
  background:#F3F7FF;
  border-radius:999px;
  font-size:13px;
  color:var(--color-neutral-900);
  box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}

/* Keywords */
.keyword {
  background:#F7F9FF;
  padding:6px 10px;
  border-radius:999px;
  font-family:monospace;
  font-size:13px;
}

/* Buttons */
.btn {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:8px 14px;
  border-radius:10px;
  cursor:pointer;
  border:none;
}
.btn-primary {
  background:var(--color-primary);
  color:white;
  box-shadow: 0 6px 18px rgba(15,98,254,0.12);
}
.btn-ghost {
  background:transparent;
  border:1px solid #E6EEF8;
  color:var(--color-neutral-900);
}

/* Accessibility: focus ring */
:focus {
  outline: 3px solid rgba(15,98,254,0.14);
  outline-offset: 3px;
  border-radius: 6px;
}

/* Responsive */
@media (max-width: 980px) {
  .app-shell {
    grid-template-columns: 1fr;
  }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# -----------------------------
# Training data and model (internal demo)
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
# XAI helper logic (preserved + polished)
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

import re  # ensure re is available after top

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
# Sidebar: brief model info + controls
# -----------------------------
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:12px'><div class='logo'>A</div><div><strong>AI Business Idea Validator</strong><div class='small'>Demo • TF-IDF + Logistic Regression</div></div></div>", unsafe_allow_html=True)
    st.caption("Design tokens: primary color, accent, neutral greys. This demo uses a small internal dataset for stability.")
    st.write("---")
    if model_report:
        with st.expander("Model details (for viva)"):
            st.write("Model: Logistic Regression on TF-IDF features.")
            st.write(f"Data Size: {len(get_training_data())} rows (internal demo set).")
            st.json(model_report, expanded=False)
    st.write("---")
    st.markdown("**Export / Handoff**")
    if st.button("Download design_tokens.json"):
        tokens = {
            "colors": {
                "primary": "#0F62FE",
                "accent": "#FF8A65",
                "neutral-900": "#101418",
                "neutral-500": "#6B7280",
                "bg-surface": "#FFFFFF"
            },
            "spacing": {"4": 16, "6": 24},
            "radius": 12
        }
        st.download_button("Download JSON", data=json.dumps(tokens, indent=2), file_name="design_tokens.json", mime="application/json")

# -----------------------------
# Main app header
# -----------------------------
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
# Main layout container
# -----------------------------
st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

# ---------- Left input column ----------
st.markdown("<div class='input-card' role='region' aria-labelledby='input-title'>", unsafe_allow_html=True)
st.markdown("<h3 id='input-title'>1️⃣ Input your idea</h3>", unsafe_allow_html=True)

# Accessible form inputs
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
        st.experimental_rerun()

# small helper links
st.markdown("<div class='small' style='margin-top:12px'>Tip: Use concrete nouns & numbers. E.g., '50 corporate clients in Mumbai' is better than 'many clients'.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close input-card

# ---------- Right results column ----------
st.markdown("<div class='results' role='region' aria-live='polite'>", unsafe_allow_html=True)

st.markdown("<div class='result-card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><div><h3 style='margin:0'>2️⃣ AI Evaluation & Insights</h3><div class='small'>Actionable guidance with model + rule-based explanations</div></div><div style='display:flex;gap:8px;align-items:center'></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# reactive area for results
results_placeholder = st.empty()

if analyze_btn:
    if not title.strip() or not desc.strip():
        st.warning("Please fill in both the title and description.")
    else:
        # show skeleton loader
        with results_placeholder.container():
            st.markdown("<div class='result-card'><div style='display:flex;gap:12px;align-items:center'><div style='flex:1'><div style='height:22px;width:70%;background:#eef3ff;border-radius:8px;margin-bottom:14px'></div><div style='height:14px;width:40%;background:#f3f5f7;border-radius:6px;margin-bottom:22px'></div><div style='height:160px;background:#fbfcfe;border-radius:10px'></div></div></div></div>", unsafe_allow_html=True)
        # simulate quick work + allow UI to update
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

        # Build DataFrames for charts
        prob_df = pd.DataFrame({"Category": classes, "Probability": np.round(proba * 100, 1)})
        feat_items = {k: v for k, v in feature_details.items() if k != "Length (words)"}
        feat_df = pd.DataFrame({"Feature": list(feat_items.keys()), "Score": list(feat_items.values())})

        # Render results (main)
        with results_placeholder.container():
            st.markdown("<div class='result-card' role='region' aria-label='Top metrics'>", unsafe_allow_html=True)
            # Top metrics: score + category
            col1, col2 = st.columns([1.6, 1])
            with col1:
                # Animated count-up for score
                score_holder = st.empty()
                score_holder.markdown(f"<div class='metric-big'><div><div class='metric-number'>{overall_score}</div><div class='small'>Overall Feasibility Score</div></div><div style='display:flex;flex-direction:column;align-items:flex-start;gap:8px'><div class='pill' style='background:#EBF4FF;color:var(--color-primary)'>Predicted: <strong style='margin-left:8px'>{pred_label}</strong></div><div class='small'>Model confidence: {int(round(max(proba)*100))}%</div></div></div>", unsafe_allow_html=True)
                # score count-up effect (progressive)
                for v in range(0, overall_score + 1, max(1, overall_score // 20 or 1)):
                    score_holder.markdown(f"<div class='metric-big'><div><div class='metric-number'>{v}</div><div class='small'>Overall Feasibility Score</div></div><div style='display:flex;flex-direction:column;align-items:flex-start;gap:8px'><div class='pill' style='background:#EBF4FF;color:var(--color-primary)'>Predicted: <strong style='margin-left:8px'>{pred_label}</strong></div><div class='small'>Model confidence: {int(round(max(proba)*100))}%</div></div></div>", unsafe_allow_html=True)
                    time.sleep(0.01)
            with col2:
                st.markdown("<div style='display:flex;flex-direction:column;gap:8px;align-items:flex-end'><button class='btn btn-ghost' onclick='window.print()'>Print report</button></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Second row: charts
            st.markdown("<div style='display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:12px'>", unsafe_allow_html=True)
            # Probabilities chart
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><strong>Class Probabilities</strong><div class='small'>Model output</div></div>", unsafe_allow_html=True)
                fig_p = px.bar(prob_df.sort_values("Probability"), x="Probability", y="Category", orientation="h", text="Probability", range_x=[0, 100])
                fig_p.update_layout(margin=dict(t=12, b=12, l=6, r=6), height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig_p.update_traces(marker_color=[ "#FFB69A" if c=="Low" else ("#FF8A65" if c=="Medium" else "#0F62FE") for c in prob_df["Category"]])
                st.plotly_chart(fig_p, use_container_width=True)
                # Export CSV
                csv_buf = prob_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download probabilities CSV", data=csv_buf, file_name="probabilities.csv", mime="text/csv")
                st.markdown("</div>", unsafe_allow_html=True)

            # Feature breakdown chart
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><strong>Feature Breakdown</strong><div class='small'>Explainable features</div></div>", unsafe_allow_html=True)
                fig_f = px.bar(feat_df.sort_values("Score"), x="Score", y="Feature", orientation="h", text="Score", range_x=[0, 100])
                fig_f.update_layout(margin=dict(t=12, b=12, l=6, r=6), height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_f, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Third row: details (keywords, models, risks, steps)
            st.markdown("<div style='display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px'>", unsafe_allow_html=True)
            # Left detail column
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<strong>Extracted keywords</strong>")
                if keywords:
                    chip_html = "<div class='chips' style='margin-top:8px'>"
                    for k in keywords:
                        chip_html += f"<div class='keyword' role='note' tabindex='0'>{k}</div>"
                    chip_html += "</div>"
                    st.markdown(chip_html, unsafe_allow_html=True)
                else:
                    st.write("_No strong keywords extracted. Try using more specific, concrete words._")
                st.markdown("<hr style='margin-top:12px;margin-bottom:12px'/>", unsafe_allow_html=True)
                st.markdown("<strong>Suggested business model(s)</strong>")
                model_chips = "<div style='margin-top:8px' class='chips'>"
                for m in models:
                    model_chips += f"<div class='chip' tabindex='0'>{m}</div>"
                model_chips += "</div>"
                st.markdown(model_chips, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Right detail column
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

            # Quick research links
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

            # Technical explanation expander (viva / handoff)
            with st.expander("Technical explanation (for report / viva)"):
                st.markdown("""
                **Model architecture**
                - TF-IDF vectorization of title + description + industry.
                - Logistic Regression classifier outputs class probabilities.
                - Rule-based XAI features (Keyword richness, Novelty, Simplicity, Industry demand) for interpretability.

                **Handoff notes (developer)**
                - Design tokens available via sidebar download.
                - Charts use Plotly; replace with your preferred charting library if needed.
                - Accessibility: form inputs include descriptive labels and help text.
                """)

# close results & app-shell wrappers
st.markdown("</div>", unsafe_allow_html=True)  # close results
st.markdown("</div>", unsafe_allow_html=True)  # close app-shell

# Footer / small credits
st.markdown("""
<div style="margin-top:18px;color:var(--color-neutral-500);font-size:13px">
Designed for founders • Demo model uses a small internal dataset • Replace demo data with your GitHub dataset for production.
</div>
""", unsafe_allow_html=True)
