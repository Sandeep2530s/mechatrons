import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ✅ Step 1: Load SMS Spam Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")  # Load dataset
df = df[['v1', 'v2']]  # Keep only necessary columns
df.columns = ['label', 'message']  # Rename columns
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels

# ✅ Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# ✅ Step 3: Convert Text to Features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Step 4: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ✅ Step 5: Save Model and Vectorizer
with open("sms_spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("sms_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("🎉 SMS Spam Model Trained and Saved Successfully!")
