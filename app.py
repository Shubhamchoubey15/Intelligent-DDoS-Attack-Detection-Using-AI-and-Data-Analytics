from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("ddos_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # Process input data (Same preprocessing as before)
    label_encoders = {
        "Source IP": LabelEncoder(),
        "Dest IP": LabelEncoder(),
        "Highest Layer": LabelEncoder(),
        "Transport Layer": LabelEncoder()
    }
    for col in ["Source IP", "Dest IP", "Highest Layer", "Transport Layer"]:
        df[col] = label_encoders[col].fit_transform(df[col])

    scaler = StandardScaler()
    df[["Packet Length", "Packets/Time"]] = scaler.fit_transform(df[["Packet Length", "Packets/Time"]])

    prediction = model.predict(df)
    return jsonify({"attack_label": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
