from flask import Flask, render_template, request
import pickle
import os
import requests

app = Flask(_name_)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

API_KEY = os.environ.get("FACT_CHECK_API_KEY")


def analyze(text):
    clean = text.strip()

    if not clean:
        return None

    # Search existing fact-check articles
    if API_KEY:
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

            params = {
                "query": clean[:500],
                "key": API_KEY
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                claims = data.get("claims", [])

                if claims:
                    rating = claims[0].get(
                        "claimReview", [{}]
                    )[0].get("textualRating", "").lower()

                    if any(word in rating for word in [
                        "false", "fake", "misleading",
                        "pants on fire", "incorrect"
                    ]):
                        return {
                            "status": "FAKE NEWS",
                            "note": "A published fact-check report rated this claim negatively."
                        }

                    if any(word in rating for [
                        "true", "correct", "accurate"
                    ]):
                        return {
                            "status": "REAL NEWS",
                            "note": "A published fact-check report rated this claim positively."
                        }

        except Exception:
            pass

    # If no matching fact-check report is found,
    # use the existing trained ML model.
    news_vector = vectorizer.transform([clean])
    prediction = model.predict(news_vector)[0]

    if prediction == 0:
        return {
            "status": "FAKE NEWS",
            "note": "The trained ML model predicted this as fake news. This is not a verified fact-check."
        }

    return {
        "status": "REAL NEWS",
        "note": "The trained ML model predicted this as real news. This is not a verified fact-check."
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


if _name_ == "_main_":
    app.run(debug=True)
