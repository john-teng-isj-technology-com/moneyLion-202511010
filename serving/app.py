from flask import Flask, request, jsonify, render_template
from pathlib import Path
import numpy as np, xgboost as xgb, json
from serving.helpers import make_feature_vector, SCHEMA, ARTIFACT_DIR

app = Flask(__name__, template_folder=str(Path(__file__).with_name("templates")))

booster = xgb.Booster()
booster.load_model(ARTIFACT_DIR / "xgb_model.json")
THR = SCHEMA.get("decision_threshold", 0.5)

@app.get("/health")
def health():
    return "ok", 200

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    rows    = payload.get("rows", [])
    mat     = np.vstack([make_feature_vector(r) for r in rows])
    probs   = booster.predict(xgb.DMatrix(mat)).tolist()
    preds   = [int(p >= THR) for p in probs]
    return jsonify({"probs": probs, "preds": preds})

@app.route("/", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        raw = request.form.to_dict()
        vec = make_feature_vector(raw).reshape(1, -1)
        prob = float(booster.predict(xgb.DMatrix(vec))[0])
        pred = int(prob >= THR)
        return render_template("result.html", prob=prob, pred=pred)
    return render_template("form.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
