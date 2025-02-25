import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 📌 Load Dataset from the Root Folder
csv_file = "./DDoS_dataset.csv"  # Ensure the file exists in the script's directory
df = pd.read_csv(csv_file)

# 📌 Load Pre-trained Models
scaler = joblib.load("scaler.pkl")  # Standard Scaler for numerical data
label_encoders = joblib.load("label_encoders.pkl")  # Label encoders for categorical data
model = joblib.load("ddos_model.pkl")  # Pre-trained Machine Learning Model

# 📌 Apply Label Encoding for Categorical Features
for col in label_encoders:
    if col in df.columns:
        df[col] = df[col].astype(str)  # Convert to string before encoding
        df[col] = label_encoders[col].transform(df[col])  # Apply encoding

# 📌 Scale Numerical Features
num_cols = getattr(scaler, "feature_names_in_", [])  # Get features used in scaling
if len(num_cols) > 0:
    df[num_cols] = scaler.transform(df[num_cols])

# 📌 Ensure Data Matches Model Input
model_features = getattr(model, "feature_names_in_", [])  # Get features used by the model
df = df.reindex(columns=model_features, fill_value=0)  # Reorder and fill missing values

# 📌 Make Predictions
df["Predicted Target"] = model.predict(df)  # 0 = Normal, 1 = DDoS Attack

# 📊 **Plot 1: Pie Chart of Traffic Distribution**
attack_counts = df["Predicted Target"].value_counts()
labels = ["Normal Traffic", "DDoS Attack"]
values = [attack_counts.get(0, 0), attack_counts.get(1, 0)]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct="%1.1f%%", colors=["green", "red"], startangle=90)
plt.title("Traffic Distribution: Normal vs DDoS")
plt.show()

# 📊 **Plot 2: Time-Series of Traffic Flow**
df["Time"] = pd.to_datetime(df.index, unit="s")  # Convert index to datetime format

plt.figure(figsize=(10, 5))
sns.lineplot(x=df["Time"], y=df["Predicted Target"])
plt.xlabel("Time")
plt.ylabel("Traffic Type (0=Normal, 1=DDoS)")
plt.title("Traffic Flow Over Time")
plt.show()

# 📊 **Plot 3: Top 10 Source IPs Involved in DDoS Attacks**
df["Source IP"] = df["Source IP"].astype(str)
attack_sources = df[df["Predicted Target"] == 1]["Source IP"].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=attack_sources.index, y=attack_sources.values, palette="Reds")
plt.xticks(rotation=45)
plt.xlabel("Source IP")
plt.ylabel("Attack Count")
plt.title("Top 10 Source IPs of DDoS Attacks")
plt.show()

# 📊 **Plot 4: Interactive Traffic Visualization (Plotly)**
fig = px.histogram(df, x="Predicted Target", color="Predicted Target",
                   title="Traffic Type Distribution",
                   labels={"Predicted Target": "0 = Normal, 1 = DDoS"},
                   color_discrete_map={0: "green", 1: "red"})
fig.show()

# 📁 **Save Predictions to a CSV File**
df.to_csv("predicted_output.csv", index=False)
print("\n✅ Predictions saved to `predicted_output.csv`")
