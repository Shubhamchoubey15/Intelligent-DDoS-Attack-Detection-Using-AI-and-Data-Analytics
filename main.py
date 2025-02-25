import pandas as pd
import joblib
import socket
import requests
import ipaddress

# ✅ Load Pre-trained Models
scaler = joblib.load("scaler.pkl")  # For scaling numeric data
label_encoders = joblib.load("label_encoders.pkl")  # For encoding categorical data
model = joblib.load("ddos_model.pkl")  # Pre-trained DDoS detection model

# ✅ Function to Convert IP Addresses to Numeric Format
def convert_ip(ip):
    """ Convert an IP address (IPv4 or IPv6) to a numerical format """
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return -1  # Assign -1 for invalid IPs

# ✅ Get Your Current Public IP
try:
    my_ip = requests.get("https://api64.ipify.org").text
except:
    my_ip = "Unknown"

# ✅ Define Test Cases (Your IP vs Simulated DDoS IP)
test_cases = [
    {"Source IP": my_ip, "Dest IP": "192.168.1.20", "Packet Length": 500, "Packets/Time": 20},  # Normal Traffic
    {"Source IP": "10.10.10.5", "Dest IP": "255.255.255.255", "Packet Length": 3000, "Packets/Time": 500}  # Simulated DDoS
]

# ✅ Convert test cases to DataFrame
df_test = pd.DataFrame(test_cases)

# ✅ Convert IP Addresses to Numeric Values
df_test["Source IP"] = df_test["Source IP"].apply(convert_ip)
df_test["Dest IP"] = df_test["Dest IP"].apply(convert_ip)

# ✅ Apply Label Encoding for Categorical Features (if necessary)
for col in label_encoders:
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str)  # Convert to string before encoding
        if df_test[col].iloc[0] in label_encoders[col].classes_:
            df_test[col] = label_encoders[col].transform(df_test[col])
        else:
            df_test[col] = -1  # Assign -1 for unknown values

# ✅ Scale Numerical Features
num_cols = getattr(scaler, "feature_names_in_", [])
if len(num_cols) > 0:
    df_test[num_cols] = scaler.transform(df_test[num_cols])

# ✅ Ensure Data Matches Model Input Format
model_features = getattr(model, "feature_names_in_", [])
df_test = df_test.reindex(columns=model_features, fill_value=0)

# ✅ Make Predictions
predictions = model.predict(df_test)

# ✅ Print Results
for i, row in df_test.iterrows():
    status = "🚨 DDoS Attack Detected! 🚨" if predictions[i] == 1 else "✅ Normal Traffic ✅"
    print(f"\n📌 Source IP: {test_cases[i]['Source IP']} → Dest IP: {test_cases[i]['Dest IP']}")
    print(f"   📦 Packet Length: {test_cases[i]['Packet Length']} bytes")
    print(f"   ⚡ Packets per Second: {test_cases[i]['Packets/Time']}")
    print(f"   🛑 Status: {status}\n")

# ✅ Save Predictions to CSV
df_test["Predicted Target"] = predictions
df_test.to_csv("predicted_output.csv", index=False)
print("\n✅ Predictions saved to `predicted_output.csv`")
