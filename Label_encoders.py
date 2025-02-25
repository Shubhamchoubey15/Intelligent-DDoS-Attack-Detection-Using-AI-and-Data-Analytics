import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = pd.DataFrame({
    "Highest Layer": ["IPv4", "IPv6", "IPv4"],
    "Transport Layer": ["TCP", "UDP", "TCP"],
    "Source IP": ["192.168.1.1", "192.168.1.2", "192.168.1.3"],
    "Dest IP": ["192.168.1.2", "192.168.1.3", "192.168.1.4"]
})

# Encode categorical features
label_encoders = {}
for col in ["Highest Layer", "Transport Layer", "Source IP", "Dest IP"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Save encoders
with open("label_encoders.pkl", "wb") as enc_file:
    pickle.dump(label_encoders, enc_file)

print("✅ Label encoders saved successfully!")
