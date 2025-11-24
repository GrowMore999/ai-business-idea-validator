import streamlit as st
import re

st.set_page_config(page_title="AI Business Idea Validator", layout="wide")

st.title("🤖 AI Business Idea Validator")
st.write(
    "This tool uses simple NLP-style analysis to score your business idea and "
    "explain **why** it is strong or weak. It breaks the idea into features "
    "like keyword richness, industry demand, novelty, and simplicity."
)

# -----------------------------
# "AI engine" configuration
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

# Common generic words – used to estimate novelty
GENERIC_WORDS = {
    "app", "website", "platform", "service", "online", "digital",
    "solution", "system", "business", "idea", "startup"
}


def preprocess(text: str) -> str:
    """Lowercase + remove punctuation (basic NLP preprocessing)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, top_k: int = 7):
    """Very simple keyword extraction based on word frequency."""
    words = text.split()
    freq = {}
    for w in words:
        if len(w) <= 3:
            continue  # ignore very short words
        freq[w] = freq.get(w, 0) + 1
    sorted_w = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_w[:top_k]]


def compute_feature_scores(title: str, desc: str, industry: str):
    """Compute individual feature scores that the 'AI' uses."""
    full_text = preprocess(title + " " + desc)
    words = full_text.split()

    # Keyword richness = how many distinct important words
    keywords = extract_keywords(full_text)
    keyword_richness = min(1.0, len(set(keywords)) / 7.0)

    # Industry demand from predefined mapping
    industry_demand = INDUSTRY_DEMAND.get(industry, INDUSTRY_DEMAND["Other"])

    # Simulated novelty: fewer generic buzzwords → higher novelty
    generic_count = sum(1 for k in keywords if k in GENERIC_WORDS)
    if len(keywords) == 0:
        novelty = 0.4
    else:
        novelty = max(0.2, 1.0 - generic_count / len(keywords))

    # Simplicity: too long = harder to execute → lower score
    word_count = len(words)
    if word_count <= 40:
        simplicity = 0.9
    elif word_count <= 120:
        simplicity = 0.7
    else:
        simplicity = 0.5

    # Convert features to final feasibility score
    feasibility = (
        0.3 * keyword_richness +
        0.3 * industry_demand +
        0.2 * novelty +
        0.2 * simplicity
    )
    final_score = int(feasibility * 100)

    # Market potential label
    if final_score >= 75:
        market = "High"
    elif final_score >= 50:
        market = "Medium"
    else:
        market = "Low"

    feature_details = {
        "Keyword richness": round(keyword_richness * 100),
        "Industry demand": round(industry_demand * 100),
        "Novelty (less generic)": round(novelty * 100),
        "Simplicity (clarity)": round(simplicity * 100),
        "Estimated word count": word_count,
    }

    return final_score, market, keywords, feature_details


def suggest_business_models(title: str, desc: str):
    text = (title + " " + desc).lower()
    models = []

    if any(w in text for w in ["subscription", "monthly", "saas"]):
        models.append("Subscription model")
    if any(w in text for w in ["marketplace", "connect", "buyers", "sellers", "listing"]):
        models.append("Marketplace model")
    if any(w in text for w in ["delivery", "on-demand", "on demand", "logistics"]):
        models.append("On-demand / delivery model")
    if any(w in text for w in ["course", "learn", "training", "tutorial"]):
        models.append("Online course / cohort model")
    if not models:
        models.append("Direct-to-consumer (D2C)")

    return models


def identify_risks(industry: str, title: str, desc: str):
    text = (title + " " + desc).lower()
    risks = []

    if industry == "Food & Beverage":
        risks.append("Perishable inventory and wastage risk.")
    if industry in ["E-commerce", "Software/SaaS", "Fintech"]:
        risks.append("High competition – need clear differentiation.")
    if "delivery" in text or "on-demand" in text:
        risks.append("Operational complexity in logistics and last-mile delivery.")
    if "subscription" in text:
        risks.append("Churn risk – customers may cancel if value is not clear.")
    if len(risks) == 0:
        risks.append("Need to validate real customer demand and willingness to pay.")

    return risks


def suggest_next_steps(score: int):
    if score >= 75:
        return [
            "Build a simple landing page and collect at least 100 signups.",
            "Interview 5–10 target customers to refine key features.",
            "Create a small MVP and test pricing."
        ]
    elif score >= 50:
        return [
            "Talk to at least 10 potential customers and validate the core problem.",
            "Narrow down the niche (start with one segment).",
            "Test ideas using a no-code prototype or mockups."
        ]
    else:
        return [
            "Clarify the exact problem and who faces it.",
            "Avoid being too generic – focus on one clear value proposition.",
            "Study 3–5 competitors and identify what you do differently."
        ]


# -----------------------------
# Streamlit UI
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("📝 Enter your business idea")

    title = st.text_input("Business Idea Title", placeholder="e.g., Local organic juice subscription")
    desc = st.text_area(
        "Describe your business idea",
        placeholder="What problem are you solving? Who is your target customer? How does your solution work?",
        height=200
    )
    industry = st.selectbox("Industry", list(INDUSTRY_DEMAND.keys()))

    analyze_button = st.button("🔍 Analyze Idea")

with right:
    st.subheader("📊 AI Evaluation")

    if analyze_button:
        if not title.strip() or not desc.strip():
            st.warning("Please enter both a title and a description to analyze.")
        else:
            score, market, keywords, feature_details = compute_feature_scores(title, desc, industry)
            models = suggest_business_models(title, desc)
            risks = identify_risks(industry, title, desc)
            steps = suggest_next_steps(score)

            # Main score display
            st.metric(label="Overall Feasibility Score", value=f"{score}/100")
            st.write(f"**Market Potential:** `{market}`")

            # Feature breakdown
            st.markdown("### 🔬 Feature Breakdown (how the AI scored your idea)")
            feat_col1, feat_col2 = st.columns(2)
            with feat_col1:
                st.write(f"- Keyword richness: **{feature_details['Keyword richness']} / 100**")
                st.write(f"- Industry demand: **{feature_details['Industry demand']} / 100**")
            with feat_col2:
                st.write(f"- Novelty (less generic): **{feature_details['Novelty (less generic)']} / 100**")
                st.write(f"- Simplicity / clarity: **{feature_details['Simplicity (clarity)']} / 100**")

            st.write(f"- Estimated length: **{feature_details['Estimated word count']} words**")

            # Keywords
            st.markdown("### 🧠 Extracted Keywords")
            if keywords:
                st.write(", ".join([f"`{k}`" for k in keywords]))
            else:
                st.write("_No strong keywords detected – try describing with more specific terms._")

            # Business model suggestions
            st.markdown("### 💼 Suggested Business Models")
            for m in models:
                st.write(f"- {m}")

            # Risks
            st.markdown("### ⚠️ Key Risks")
            for r in risks:
                st.write(f"- {r}")

            # Next steps
            st.markdown("### ✅ Recommended Next Steps")
            for s in steps:
                st.write(f"- {s}")

            # Technical explanation for viva / teacher
            with st.expander("🔍 Show technical / AI explanation (for viva / teacher)"):
                st.markdown(
                    """
                    **How the AI-style scoring works:**

                    1. **Preprocessing (NLP):**  
                       - Converts text to lowercase  
                       - Removes punctuation and extra spaces  

                    2. **Keyword Extraction:**  
                       - Counts word frequencies  
                       - Ignores very short words  
                       - Picks the top keywords as important signals  

                    3. **Feature Engineering:**  
                       - **Keyword richness:** more distinct keywords → better score  
                       - **Industry demand:** each industry has a base demand score  
                       - **Novelty:** if keywords are too generic (e.g., *app, platform, solution*), novelty score decreases  
                       - **Simplicity:** very long ideas are penalized for complexity  

                    4. **Scoring Formula:**  
                       The final score is a weighted combination:  

                       `score = 0.3 * keyword_richness + 0.3 * industry_demand + 0.2 * novelty + 0.2 * simplicity`  

                       Then multiplied by 100 and mapped to **High / Medium / Low** market potential.
                    """
                )
    else:
        st.info("Enter your idea on the left and click **Analyze Idea** to see the AI breakdown here.")
