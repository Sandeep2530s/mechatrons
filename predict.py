import sys
import pickle
import re
import numpy as np
from scipy.sparse import hstack

sys.stdout.reconfigure(encoding='utf-8')  # ✅ Ensure correct encoding

print("📌 predict.py script started", flush=True)

# ✅ Step 1: Define URL Cleaning Function
def clean_url(url):
    url = url.lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    url = re.sub(r"[^a-zA-Z0-9\-_.@]", " ", url)
    url = re.sub(r"\s+", " ", url).strip()
    return url

# ✅ Step 2: Load Model and Vectorizer
try:
    print("📌 Loading Model...", flush=True)
    with open("phishing_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("✅ Model and vectorizer loaded successfully.", flush=True)

except Exception as e:
    print(f"❌ Error loading model: {str(e)}", flush=True)
    sys.exit(1)

# ✅ Step 3: Get Input URL
if len(sys.argv) < 2:
    print("❌ Error: No URL provided", flush=True)
    sys.exit(1)

url = sys.argv[1]
print(f"📌 Received URL: {url}", flush=True)

# ✅ Step 4: Apply Preprocessing
cleaned_url = clean_url(url)
print(f"🔄 Processing: {cleaned_url}", flush=True)

# ✅ Step 5: Convert URL to Features
try:
    print("📌 Vectorizing URL...", flush=True)
    url_tfidf = vectorizer.transform([cleaned_url])

    # Extract Additional Features
    url_features = np.array([[len(cleaned_url), cleaned_url.count('.'), cleaned_url.count('-')]])

    # ✅ Combine Features
    url_combined = hstack([url_tfidf, url_features])

    print("📌 Running Model Prediction...", flush=True)
    prediction = model.predict(url_combined)[0]

    print("✅ Prediction Completed.", flush=True)
    print("1" if prediction == 1 else "0", flush=True)

except Exception as e:
    print(f"❌ Prediction Error: {str(e)}", flush=True)
    sys.exit(1)
