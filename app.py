# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# optional imports that may be missing at runtime:
try:
    from wordcloud import WordCloud, STOPWORDS
    _HAS_WORDCLOUD = True
except Exception:
    _HAS_WORDCLOUD = False

# Set page config
st.set_page_config(page_title="AI Echo — Sentiment Insights", layout="wide")

# ---------- CONFIG ----------
DATA_PATH = "final_processed_reviews.csv"   # change if different
DATE_COL = "date"                           # your date column name
TEXT_COL = "review"                         # raw review text column
CLEAN_COL = "cleaned_review"                # cleaned text (optional)
FINAL_COL = "final_text"                    # tokenized/lemmatized text (optional)
SENTIMENT_COL = "sentiment"                 # Positive/Neutral/Negative
# ----------------------------

# ---------- helper funcs ----------
@st.cache_data
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Data file not found: `{path}`. Put your CSV in the app folder or change DATA_PATH.")
        return None
    # strip column whitespace
    df.columns = df.columns.str.strip()
    return df

def ensure_date(df, col=DATE_COL):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=False)
    return df

def safe_column(df, name, default=""):
    if name not in df.columns:
        df[name] = default
    return df

def wordcloud_image(text, title=None, width=800, height=400):
    if not _HAS_WORDCLOUD:
        st.warning("wordcloud not installed: `pip install wordcloud` to see word clouds.")
        st.write("Top words preview:")
        st.write(" ".join(str(text).split()[:200]))
        return
    wc = WordCloud(width=width, height=height, background_color="white",
                   stopwords=STOPWORDS).generate(str(text))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title)
    st.pyplot(fig)

def top_terms_bar(text, top_n=20):
    import re
    from collections import Counter
    toks = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower()).split()
    # basic stopwords set (extend as needed)
    stop = set(["the","and","you","with","for","not","this","that","are","have","was","but","they","from","your","chatgpt","openai"])
    toks = [t for t in toks if t not in stop and len(t)>2]
    wc = Counter(toks).most_common(top_n)
    if not wc:
        st.write("No tokens found.")
        return
    dfw = pd.DataFrame(wc, columns=["word","count"])
    fig = px.bar(dfw, x="word", y="count", title=f"Top {top_n} words")
    st.plotly_chart(fig, use_container_width=True)

# ---------- UI: sidebar navigation ----------
st.sidebar.title("AI Echo — Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "1. Overall sentiment",
    "2. Sentiment by rating",
    "3. Keywords per sentiment",
    "4. Sentiment over time",
    "5. Verified vs Non-verified",
    "6. Platform comparison",
    "7. Location comparison",
    "8. Review length vs rating",
    "9. Version comparison",
    "10. Negative themes",
    "11. Predict Sentiment"

])

# ---------- Load data ----------
df = load_data()
if df is None:
    st.stop()

# make sure columns exist (avoid crashes)
safe_column(df, TEXT_COL, "")
ensure_date(df, DATE_COL)
safe_column(df, "rating", np.nan)
safe_column(df, "helpful_votes", 0)
safe_column(df, "platform", "Unknown")
safe_column(df, "location", "Unknown")
safe_column(df, "version", "Unknown")
safe_column(df, "verified_purchase", "Unknown")
safe_column(df, CLEAN_COL, "")
safe_column(df, FINAL_COL, "")
safe_column(df, SENTIMENT_COL, "")

# convenience: derived fields
if DATE_COL in df.columns:
    df = df.sort_values(by=DATE_COL)
    try:
        df["month"] = df[DATE_COL].dt.to_period("M").astype(str)
    except Exception:
        df["month"] = np.nan
else:
    df["month"] = np.nan

# ---------- Home ----------
if page == "Home":
    st.title("📘 AI Echo — Sentiment Insights")
    st.markdown("""
    **This dashboard contains 11 analysis pages.**
    
    - Home: this page.  
    - Each page corresponds to one of your key questions (select from sidebar).
    
    **Instructions**
    1. Make sure `final_processed_reviews.csv` (or change `DATA_PATH`) is in this folder.  
    2. `cleaned_review` should contain cleaned text; if not present the app will use `review`.  
    3. `sentiment` should contain `Positive`/`Neutral`/`Negative`. If not, the app will infer from rating (4-5 Pos, 3 Neu, 1-2 Neg).
    """)
    st.write("Dataset preview (first 100 rows):")
    st.dataframe(df.head(100))

# ---------- 1. Overall sentiment ----------
elif page == "1. Overall sentiment":
    st.header("1️⃣ Overall sentiment of user reviews")
    if SENTIMENT_COL not in df.columns or df[SENTIMENT_COL].isna().all():
        st.info("No `sentiment` column found — inferring from rating (4-5 → Positive; 3 → Neutral; 1-2 → Negative).")
        def label_sentiment(r):
            try:
                r = float(r)
            except:
                return "Neutral"
            if r >= 4: return "Positive"
            if r == 3: return "Neutral"
            return "Negative"
        df["sentiment_inferred"] = df["rating"].apply(label_sentiment)
        use_col = "sentiment_inferred"
    else:
        use_col = SENTIMENT_COL

    counts = df[use_col].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, title="Sentiment proportions")
    st.plotly_chart(fig, use_container_width=True)
    st.write(counts)

# ---------- 2. Sentiment by rating ----------
elif page == "2. Sentiment by rating":
    st.header("2️⃣ How does sentiment vary by rating?")
    st.write("Stacked counts: rating vs sentiment")
    if SENTIMENT_COL in df.columns and not df[SENTIMENT_COL].isna().all():
        chart_col = SENTIMENT_COL
    else:
        chart_col = "sentiment_inferred" if "sentiment_inferred" in df.columns else SENTIMENT_COL
    fig = px.histogram(df, x="rating", color=chart_col, barmode="group",
                       category_orders={"rating":[1,2,3,4,5]}, title="Rating vs Sentiment")
    st.plotly_chart(fig, use_container_width=True)
    if chart_col in df.columns:
        st.write(df.groupby(["rating", chart_col]).size().unstack(fill_value=0))

# ---------- 3. Keywords per sentiment ----------
elif page == "3. Keywords per sentiment":
    st.header("3️⃣ Keywords / phrases most associated with each sentiment")
    chosen = st.selectbox("Choose sentiment", ["Positive","Neutral","Negative"])
    text_source = CLEAN_COL if (CLEAN_COL in df.columns and df[CLEAN_COL].astype(bool).sum()>0) else TEXT_COL
    sample_text = " ".join(df[df[SENTIMENT_COL]==chosen][text_source].dropna().astype(str)) if SENTIMENT_COL in df.columns else ""
    if not sample_text:
        st.warning(f"No text available for sentiment `{chosen}`. Check `{text_source}` and `{SENTIMENT_COL}`.")
    else:
        st.subheader("Word Cloud")
        wordcloud_image(sample_text, title=f"Top words — {chosen}")
        st.subheader("Top terms (bar chart)")
        top_terms_bar(sample_text, top_n=25)


# ---------- 4. Sentiment over time ----------
elif page == "4. Sentiment over time":
    st.header("4️⃣ Sentiment trend over time")
    if DATE_COL not in df.columns or df[DATE_COL].isna().all():
        st.error(f"Date column `{DATE_COL}` missing or all NaT. Ensure your dataframe has a valid `{DATE_COL}`.")
    else:
        # pick sentiment column
        sentiment_col = SENTIMENT_COL if SENTIMENT_COL in df.columns and df[SENTIMENT_COL].astype(bool).any() else None
        freq = st.radio("Group by", ["Month","Week","Day"], index=0)
        if freq == "Month":
            grp = df.groupby([pd.Grouper(key=DATE_COL, freq="M"), sentiment_col if sentiment_col else "rating"]).size().reset_index(name="count")
        elif freq == "Week":
            grp = df.groupby([pd.Grouper(key=DATE_COL, freq="W"), sentiment_col if sentiment_col else "rating"]).size().reset_index(name="count")
        else:
            grp = df.groupby([pd.Grouper(key=DATE_COL, freq="D"), sentiment_col if sentiment_col else "rating"]).size().reset_index(name="count")

        # Normalize column name for plotting
        grp[DATE_COL] = pd.to_datetime(grp[DATE_COL])
        color_col = sentiment_col if sentiment_col else "rating"
        fig = px.line(grp, x=DATE_COL, y="count", color=color_col, title=f"Sentiment counts per {freq.lower()}")
        st.plotly_chart(fig, use_container_width=True)


# ---------- 5. Verified vs Non-verified ----------
elif page == "5. Verified vs Non-verified":
    st.header("5️⃣ Do verified users leave more positive reviews?")
    if "verified_purchase" not in df.columns:
        st.warning("No `verified_purchase` column — check your CSV.")
    else:
        st.write("Average rating & sentiment by verified flag")
        avg_by_verified = df.groupby("verified_purchase")["rating"].agg(["mean","count"]).reset_index()
        st.dataframe(avg_by_verified)
        fig = px.histogram(df, x="verified_purchase", color=SENTIMENT_COL if SENTIMENT_COL in df.columns else "rating",
                           barmode="group", title="Verified purchase vs sentiment/rating")
        st.plotly_chart(fig, use_container_width=True)

# ---------- 6. Platform comparison ----------
elif page == "6. Platform comparison":
    st.header("6️⃣ Platform (Web vs Mobile) comparison")
    st.write("Average rating per platform")
    avg_platform = df.groupby("platform")["rating"].mean().reset_index().sort_values("rating", ascending=False)
    st.dataframe(avg_platform)
    fig = px.bar(avg_platform, x="platform", y="rating", title="Average rating by platform")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 7. Location comparison ----------
elif page == "7. Location comparison":
    st.header("7️⃣ Location-based sentiment")
    top_n = st.slider("Top N locations to show", 5, 20, 10)
    loc_counts = df["location"].value_counts().head(top_n).index
    df_loc = df[df["location"].isin(loc_counts)]
    fig = px.histogram(df_loc, x="location", color=SENTIMENT_COL if SENTIMENT_COL in df.columns else "rating",
                       barmode="group", title=f"Top {top_n} locations by sentiment")
    st.plotly_chart(fig, use_container_width=True)
    # show table of counts by sentiment (safe)
    if SENTIMENT_COL in df.columns:
        st.write(df_loc.groupby(["location", SENTIMENT_COL]).size().unstack(fill_value=0))
    else:
        st.write(df_loc.groupby(["location", "rating"]).size().unstack(fill_value=0))

# ---------- 8. Review length vs rating ----------
elif page == "8. Review length vs rating":
    st.header("8️⃣ Are longer reviews more likely to be negative or positive?")
    # ensure review_length exists or compute (words)
    if "review_length" not in df.columns or df["review_length"].isna().all():
        df["review_length"] = df[TEXT_COL].astype(str).apply(lambda x: len(x.split()))
    fig = px.box(df, x=SENTIMENT_COL if SENTIMENT_COL in df.columns else "rating", y="review_length",
                 title="Review length distribution by sentiment/rating")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Average review length by sentiment/rating:")
    st.dataframe(df.groupby(SENTIMENT_COL if SENTIMENT_COL in df.columns else "rating")["review_length"].mean().reset_index())

# ---------- 9. Version comparison ----------
elif page == "9. Version comparison":
    st.header("9️⃣ Which app versions have better sentiment?")
    if "version" not in df.columns:
        st.warning("No `version` column found.")
    else:
        avg_ver = df.groupby("version")["rating"].mean().reset_index().sort_values("rating", ascending=False)
        st.dataframe(avg_ver.head(30))
        fig = px.bar(avg_ver.head(30), x="version", y="rating", title="Average rating by version (top 30)")
        st.plotly_chart(fig, use_container_width=True)

# ---------- 10. Negative themes ----------
elif page == "10. Negative themes":
    st.header("🔟 Most common negative themes")
    text_source = CLEAN_COL if (CLEAN_COL in df.columns and df[CLEAN_COL].astype(bool).sum()>0) else TEXT_COL
    negative_text = " ".join(df[df[SENTIMENT_COL]=="Negative"][text_source].dropna().astype(str)) if SENTIMENT_COL in df.columns else ""
    if not negative_text:
        st.warning("No negative reviews found or `sentiment` not labeled.")
    else:
        st.subheader("Word Cloud for negative reviews")
        wordcloud_image(negative_text, title="Negative review word cloud")
        st.subheader("Top terms in negative reviews")
        top_terms_bar(negative_text, top_n=30)



        
 # ---------- 11. Predict Sentiment ----------
elif page == "11. Predict Sentiment":
    st.header("📝 Predict Sentiment for New Reviews")

    st.write("Type any user review below and get **Positive / Neutral / Negative** sentiment prediction.")

    # Text input
    user_input = st.text_area("Enter a review to analyze:", height=150)

    # Load your trained model
    import pickle

    try:
        with open("sentiment_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except:
        st.error("Model files not found! Please keep `sentiment_model.pkl` and `tfidf_vectorizer.pkl` in the app folder.")
        st.stop()

    # Predict button
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            # Transform
            X = vectorizer.transform([user_input])
            pred = model.predict(X)[0]

            # Display result
            if pred == "Positive":
                st.success("🎉 Sentiment: **Positive**")
            elif pred == "Neutral":
                st.info("😐 Sentiment: **Neutral**")
            else:
                st.error("⚠️ Sentiment: **Negative**")

                # ---------- 11. Predict Sentiment ----------
elif page == "11. Predict Sentiment":
    st.header("🔮 11. Predict Sentiment from User Review")

    st.markdown("""
    Enter any text and the model will classify it as:
    - 👍 Positive  
    - 😐 Neutral  
    - 👎 Negative  
    """)

    # Load saved model & vectorizer
    import pickle
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Load files
    try:
        with open("sentiment_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except:
        st.error("❌ Model files not found! Keep 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' in the app folder.")
        st.stop()

    # Preprocessing functions (same as training)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z ]", " ", text)
        text = " ".join(text.split())
        return text

    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    # User input
    user_text = st.text_area("Enter a review to analyze:", height=150)

    if st.button("Predict Sentiment"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            # Clean + preprocess
            cleaned = clean_text(user_text)
            final_text = preprocess(cleaned)

            # Vectorize
            vec = vectorizer.transform([final_text])

            # Predict
            prediction = model.predict(vec)[0]

            # Display result
            if prediction == "Positive":
                st.success("🌟 **Sentiment: Positive**")
            elif prediction == "Neutral":
                st.info("😐 **Sentiment: Neutral**")
            else:
                st.error("👎 **Sentiment: Negative**")



# ---------- footer ----------
st.markdown("---")
st.caption("AI Echo — Sentiment Dashboard. Change DATA_PATH at top of app.py if your file name differs.")

