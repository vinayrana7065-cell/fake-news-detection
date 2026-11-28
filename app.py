from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained ML model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def analyze(text):
    clean = text.strip()

    if not clean:
        return None

    # Convert news text into ML features
    news_vector = vectorizer.transform([clean])

    # Predict news label
    prediction = model.predict(news_vector)[0]

    # Get prediction confidence
    probabilities = model.predict_proba(news_vector)[0]
    confidence = round(max(probabilities) * 100, 2)

    if prediction == 0:
        status = "Likely Fake News"
        note = "The trained model classified this text as fake news. This is a prediction, not a verified fact-check."
    else:
        status = "Likely Real News"
        note = "The trained model classified this text as real news. This is a prediction, not a verified fact-check."

    return {
        "score": confidence,
        "status": status,
        "note": note,
        "hits": []
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    submitted = ""

    if request.method == "POST":
        submitted = request.form.get("news_text", "")
        result = analyze(submitted)

    return render_template(
        "index.html",
        result=result,
        submitted=submitted
    )


if __name__ == "__main__":
    app.run(debug=True)
