import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import json

# Load dataset
df = pd.read_csv("dataset.csv")
df.dropna(inplace=True)

# Features and labels
X = df['text']
y = df['emotion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model
with open("emotion_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

# Save accuracy to a separate file
with open("accuracy.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print("✅ Model and accuracy saved successfully.")

