# NLP-Sentiment-Analysis
Sentiment analysis is used to identify whether user reviews are Positive, Neutral, or Negative.
This project extracts insights from customer reviews, highlights major themes, and helps improve customer experience.

Project Features

A. Data Preprocessing 

Removal of missing values
Dropping unwanted columns (username, review_length, etc.)
Text cleaning
Lowercasing
Removing special characters
Removing stopwords
Lemmatization
Date conversion and sorting
Creating new columns:
year, month, week 

B. Exploratory Data Analysis (EDA)

Sentiment distribution
Review ratings distribution
Rating vs Sentiment comparison
Sentiment over time
Word clouds for Positive/Negative reviews
Top keywords per sentiment
Verified vs Non-Verified review patterns
Platform comparison (Web vs Android)
Location-based analysis
Review length vs rating
Version comparison
Negative themes extraction
A total of 10 dashboards were created. 

Machine Learning Model

Model: Logistic Regression
Vectorizer: TF-IDF
Labels: Positive / Neutral / Negative
sentiment_model.pkl
tfidf_vectorizer.pkl
These are used in the Streamlit app to predict new user review sentiment. 

Streamlit Dashboard (app.py)

Home page

Overall Sentiment
Sentiment by Rating
Keywords per Sentiment
Sentiment Over Time
Verified vs Non-verified
Platform Comparison
Location Comparison
Review Length vs Rating
Negative Themes
Predict Sentiment (ML model integrated)

Users can:

View visual insights using Plotly charts
See word-clouds
Select filters
Enter any review and get instant sentiment prediction
