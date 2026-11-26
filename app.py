from flask import Flask, render_template, request
import re

app = Flask(__name__)

# Lightweight demo classifier: illustrative keyword/pattern scoring, NOT a trained fact-checker.
# Replace this with a verified dataset + ML model for a research-grade project.

MISLEADING_PATTERNS = [
    r'you won', r'claim your prize', r'100% guaranteed', r'click here', r'shocking',
    r'breaking.*secret', r'no one is telling', r' miracle', r'cure.*overnight',
    r'forward this to everyone', r'urgent!!!', r'guaranteed profit', r'free money'
]

CAUTION_PATTERNS = [r'!{2,}', r'\?{2,}', r'\b(never|always|everyone|nobody)\b', r'\b(viral|secret|miracle)\b']


def analyze(text):
    clean = text.strip()
    if not clean:
        return None
    score = 0
    hits = []
    lower = clean.lower()
    for pattern in MISLEADING_PATTERNS:
        if re.search(pattern, lower):
            score += 18
            hits.append(pattern.replace('.*', ' '))
    for pattern in CAUTION_PATTERNS:
        if re.search(pattern, lower):
            score += 7
    # Very short text provides little context.
    if len(clean.split()) < 8:
        score += 12
    score = min(score, 95)
    status = 'Potentially misleading' if score >= 35 else 'No obvious warning patterns'
    note = ('This text contains patterns often seen in sensational or misleading posts.'
            if score >= 35 else 'This demo found few warning patterns in the text.')
    return {'score': score, 'status': status, 'note': note, 'hits': hits}


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    submitted = ''
    if request.method == 'POST':
        submitted = request.form.get('news_text', '')
        result = analyze(submitted)
    return render_template('index.html', result=result, submitted=submitted)


if __name__ == '__main__':
    app.run(debug=True)
