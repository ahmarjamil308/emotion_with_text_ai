from flask import Flask, render_template, request
import pickle
import json

app = Flask(__name__)

# Load model and vectorizer
with open("emotion_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Load model accuracy from JSON
try:
    with open("accuracy.json", "r") as f:
        accuracy_data = json.load(f)
        model_accuracy = round(accuracy_data.get("accuracy", 0) * 100, 2)
except:
    model_accuracy = "N/A"

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = ""
    confidence = ""
    if request.method == "POST":
        text = request.form["text"]
        vec = vectorizer.transform([text])
        pred_probs = model.predict_proba(vec)[0]     # Get all class probabilities
        emotion = model.predict(vec)[0]              # Get top class
        confidence = round(max(pred_probs) * 100, 2) # Get confidence percentage

    return render_template("index.html", emotion=emotion, confidence=confidence, accuracy=model_accuracy)


if __name__ == "__main__":
    app.run(debug=True)
