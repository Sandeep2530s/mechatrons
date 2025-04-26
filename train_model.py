import pandas as pd
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report

# ✅ Step 1: Load Dataset
try:
    df = pd.read_csv(r"D:\Mini-Project-Using AI\node\phishing_email/emails.csv")  # Ensure correct filename
    print("✅ Dataset loaded successfully")
except FileNotFoundError:
    print("❌ Error: Dataset file not found.")
    exit()

# ✅ Step 2: Ensure Dataset Has Both Phishing (1) and Safe (0) URLs
if 'label' not in df.columns or 'url' not in df.columns:
    print("❌ Error: Missing 'label' or 'url' column in dataset.")
    exit()

# ✅ Step 3: Preprocess URLs
def clean_url(url):
    url = url.lower()
    url = re.sub(r"https?://", "", url)  # Remove http:// or https://
    url = re.sub(r"www\.", "", url)  # Remove www.
    url = re.sub(r"[^a-zA-Z0-9\-_.@]", " ", url)  # Keep phishing indicators
    url = re.sub(r"\s+", " ", url).strip()
    return url

df['clean_url'] = df['url'].apply(clean_url)

# ✅ Step 4: Balance Dataset by Undersampling Phishing URLs
df_safe = df[df['label'] == 0]
df_phish = df[df['label'] == 1].sample(n=len(df_safe), random_state=42)  # Match phishing count to safe count

df_balanced = pd.concat([df_safe, df_phish], ignore_index=True)

# ✅ Step 5: Split Data for Training
X_train, X_test, y_train, y_test = train_test_split(df_balanced['clean_url'], df_balanced['label'], test_size=0.2, random_state=42)

# ✅ Step 6: Convert URLs to TF-IDF Features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data

# ✅ Step 7: Extract Additional Features
def extract_features(X):
    return np.array([[len(url), url.count('.'), url.count('-')] for url in X])  # Add dot and hyphen count

feature_transformer = FunctionTransformer(extract_features, validate=False)
X_train_features = feature_transformer.transform(X_train)
X_test_features = feature_transformer.transform(X_test)

# ✅ Step 8: Combine All Features
X_train_combined = hstack([X_train_tfidf, X_train_features])
X_test_combined = hstack([X_test_tfidf, X_test_features])

# ✅ Step 9: Train the Model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train_combined, y_train)

# ✅ Step 10: Evaluate Model
y_pred = model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# ✅ Step 11: Save Model and Vectorizer
with open("phishing_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("🎉 Training complete! 'phishing_model.pkl' and 'vectorizer.pkl' saved successfully.")
