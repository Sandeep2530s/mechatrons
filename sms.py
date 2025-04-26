import sys
import pickle
sys.stdout.reconfigure(encoding='utf-8')  # Fix Unicode issues in Windows

# ✅ Step 1: Load Model & Vectorizer
try:
    with open("sms_spam_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("sms_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("✅ SMS Spam Model Loaded Successfully.", flush=True)

except Exception as e:
    print(f"❌ Error loading model: {str(e)}", flush=True)
    sys.exit(1)

# ✅ Step 2: Get SMS Message Input
if len(sys.argv) < 2:
    print("❌ Error: No SMS message provided", flush=True)
    sys.exit(1)

message = sys.argv[1].strip()

if not message:
    print("❌ Error: Empty SMS message received", flush=True)
    sys.exit(1)

print(f"📌 Received SMS: {message}", flush=True)

try:
    # ✅ Step 3: Vectorize & Predict
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]

    print("✅ Prediction Completed.", flush=True)
    print("1" if prediction == 1 else "0", flush=True)

except Exception as e:
    print(f"❌ Error processing SMS: {str(e)}", flush=True)
    sys.exit(1)
