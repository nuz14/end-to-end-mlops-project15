
from flask import Flask, request, jsonify
from prometheus_client import Counter, generate_latest
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")
names = ["setosa", "versicolor", "virginica"]
requests_count = Counter("prediction_requests", "Total requests")

@app.route("/")
def home():
    return "MLOps API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    x = request.json["features"]
    if len(x) != 4:
        return jsonify({"error": "Enter exactly four features"}), 400
    requests_count.inc()
    p = int(model.predict([x])[0])
    return jsonify({"prediction": p, "flower": names[p]})

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": "text/plain; version=0.0.4"
    }

app.run(host="0.0.0.0", port=8000)
