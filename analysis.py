import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load your processed dataset
df = pd.read_csv("final_processed_reviews.csv")

# Use your cleaned text column or raw review text
text = df["final_text"]  # if exists
# text = df["cleaned_review"]  # choose your column
# text = df["review"]          # fallback

labels = df["sentiment"]  # Positive / Neutral / Negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Save model
pickle.dump(model, open("sentiment_model.pkl", "wb"))

# Save vectorizer
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
