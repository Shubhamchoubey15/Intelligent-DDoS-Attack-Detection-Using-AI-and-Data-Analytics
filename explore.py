import pandas as pd
import numpy as np
import joblib  # For saving model, encoders, and scaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ✅ Load dataset
df = pd.read_csv("DDoS_dataset.csv")  

# ✅ Encode categorical variables
label_encoders = {}
for col in ['Source IP', 'Dest IP', 'Highest Layer', 'Transport Layer']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for future use

# ✅ Split data into features and target
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target

# ✅ Normalize numeric features
scaler = StandardScaler()
X[['Packet Length', 'Packets/Time']] = scaler.fit_transform(X[['Packet Length', 'Packets/Time']])

# ✅ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\n✅ Model Cross-Validation Complete!\nCross-Validation Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# ✅ Predictions & Evaluation
y_pred = model.predict(X_test)
print("\n✅ Model Training Complete!")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save the trained model, encoders, and scaler
joblib.dump(model, "ddos_model.pkl")
print("\n💾 Model saved as 'ddos_model.pkl'")

joblib.dump(label_encoders, "label_encoders.pkl")
print("💾 Label Encoders saved as 'label_encoders.pkl'")

joblib.dump(scaler, "scaler.pkl")
print("💾 Scaler saved as 'scaler.pkl'")
