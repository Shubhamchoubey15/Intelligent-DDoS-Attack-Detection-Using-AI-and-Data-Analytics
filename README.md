## **Intelligent DDoS Attack Detection Using AI and Data Analytics**  

### **📌 Project Overview**  
This project focuses on detecting **DDoS (Distributed Denial of Service) attacks** using **Machine Learning and Data Analytics**. The model has been trained on an **open-source, synthetic (fake) dataset** and can be used to classify network traffic as **normal or DDoS**.  

---

### **📂 Files in This Repository**  
✔ **`DDoS_dataset.csv`** → Open-source dataset used for training.  
✔ **`main.py`** → Python script for data preprocessing, model loading, and attack detection.  
✔ **`scaler.pkl`** → Pre-trained scaler for feature normalization.  
✔ **`label_encoders.pkl`** → Label encoders for categorical feature transformation.  
✔ **`ddos_model.pkl`** → Pre-trained ML model for attack classification.  

---

### **🔧 Technologies & Frameworks Used**  
- **Programming Language**: Python 🐍  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly  
- **Machine Learning**: Random Forest (Used in this project) / Decision Tree / XGBoost (Users can experiment with different models)  
- **Visualization**: Seaborn, Plotly, Matplotlib  

---

### **🚀 How to Use This Project?**  
1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/Shubhamchoubey15/Intelligent-DDoS-Attack-Detection-Using-AI-and-Data-Analytics.git
cd Intelligent-DDoS-Attack-Detection-Using-AI-and-Data-Analytics

```
  
2️⃣ **Install Required Libraries**  
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn plotly
```
  
3️⃣ **Run the Attack Detection Script**  
```bash
python app.py  # Module loads and checks for errors  
python Main.py  # Run tests on the model  
python Chart_view.py  # Visual analysis and real-time tracking  

```

---

### **📊 Features in the Dataset**  
The dataset contains multiple network traffic features used to detect DDoS attacks. Key features include:  

- **Source IP & Destination IP** → Identifies traffic origins  
- **Packet Length, Duration, Protocol** → Analyzes network behavior  
- **Bytes Sent, Packets Sent** → Detects abnormal traffic spikes  
- **Attack Labels (0 = Normal, 1 = DDoS)**  

---

### **💡 Future Scope**  
✔ Improve detection using **deep learning models** (LSTMs, CNNs).  
✔ Expand dataset with **real-world attack data**.  
✔ Deploy as an **API for real-time attack detection**.  

---

### **📌 Disclaimer**  
- The dataset used is **open-source and synthetic** (not real-world attack data).  
- This project is for **educational and research purposes** only.  
- Users can **modify, train, and use** the model as needed.

- ### 🎥 Live Attack Detection Demo 
- ![WhatsApp Image 2025-02-26 at 01 18 34_4169e5e6](https://github.com/user-attachments/assets/579e5e93-2300-4d17-b714-6d588a3e374e)
![Screenshot (306)](https://github.com/user-attachments/assets/66578483-f487-4717-aa48-e4fac282d086)
![Screenshot (308)](https://github.com/user-attachments/assets/fcc67ddd-db1f-4c11-b361-ab064a288679)
![Screenshot (310)](https://github.com/user-attachments/assets/986b6ca9-7461-4f58-9552-85557e6ffbf5)

![image](https://github.com/user-attachments/assets/0784b6fa-f818-419a-b44b-b47f390bcf2f)
