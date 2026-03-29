from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    transformed = vectorizer.transform([news])
    prediction = model.predict(transformed)[0]

    if prediction == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)