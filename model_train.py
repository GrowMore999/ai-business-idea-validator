import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample training dataset (you can expand later)
data = {
    "idea": [
        "subscription based organic juice delivery",
        "generic app for everything",
        "platform to connect tutors and students",
        "ai powered medical diagnosis assistant",
        "local handmade crafts ecommerce"
    ],
    "label": ["High", "Low", "Medium", "High", "Medium"]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["idea"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved!")
