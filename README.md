 
# 🚀 Wireless Capsule Endoscopy (WCE) Image Classification | KAUHC Dataset  

This project leverages **deep learning** techniques to classify **Wireless Capsule Endoscopy (WCE) images** using the **King Abdulaziz University Hospital Capsule (KAUHC) dataset**. The dataset consists of **3301 labeled images** categorized into:  
✔️ **Normal**  
✔️ **Arteriovenous Malformations (AVM)**  
✔️ **Ulcer**  

The goal is to develop an **automated diagnostic system** for small-bowel abnormalities, supporting **AI-driven healthcare solutions**.  

---

## 📂 Dataset Overview  

- **Dataset Name:** KAUHC (King Abdulaziz University Hospital Capsule)  
- **Total Images:** 3301  
- **Categories:** Normal, AVM, Ulcer  
- **Source:** Wireless Capsule Endoscopy (WCE) studies from Saudi Arabian patients  
- **Annotation:** Labeled by multiple gastroenterologists  

🔗 **If using this dataset for research, please cite:**  
Ghandorh, H., Bali, H. H., Yafooz, W. M. S., Boulila, W., & Alsahafi, M. (2024).  

---

## 🛠️ Implementation Steps  

### 1️⃣ Data Loading & Preprocessing  
- Load images and apply **data augmentation** (rotation, flipping, normalization).  
- Convert images to **numpy arrays** for deep learning compatibility.  
- Split the dataset into **training, validation, and testing** sets.  

### 2️⃣ Data Visualization  
- Display sample images using **Matplotlib**.  
- Plot **image distribution per class** to understand dataset balance.  

### 3️⃣ Model Definition  
- Build a **Convolutional Neural Network (CNN)** using TensorFlow/Keras.  
- Utilize **Transfer Learning** (e.g., EfficientNet, ResNet) for better accuracy.  
- Apply **Batch Normalization** & **Dropout** for regularization.  

### 4️⃣ Model Training & Callbacks  
- Use **Early Stopping** to prevent overfitting.  
- Implement **ModelCheckpoint** to save the best model.  
- Train the model with **Adam optimizer & categorical cross-entropy loss**.  

### 5️⃣ Model Evaluation  
- Plot **accuracy/loss curves** for training & validation.  
- Generate a **confusion matrix & classification report**.  
- Compute key **metrics (Precision, Recall, F1-score, ROC-AUC)**.  

### 6️⃣ Predictions on New Data  
- Load new WCE images and predict their class using the trained model.  
- Visualize results with **Grad-CAM heatmaps** to interpret CNN decisions.  

---

## 📌 Results & Insights  
✅ Achieved **high classification accuracy** on endoscopic images.  
✅ **Data augmentation** significantly improved generalization.  
✅ Transfer learning models outperformed standard CNNs.  
✅ Potential applications in **AI-driven medical diagnosis**.  

---

## 📜 How to Run This Project  

1️⃣ Clone the repository:  
```bash
git clone https://github.com/your-username/WCE-Image-Classification.git
cd WCE-Image-Classification
```

2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```

3️⃣ Run the model training script:  
```bash
python train.py
```

4️⃣ Test on new images:  
```bash
python predict.py --image path/to/image.jpg
```

---

## 🚀 Future Work  
- Fine-tuning **transformer-based vision models** (ViTs).  
- Expanding dataset size for **better generalization**.  
- Integrating this model into a **real-time medical diagnostic app**.  

---

## 📎 Resources  
🔗 **Kaggle Notebook:** [https://www.kaggle.com/code/arifmia/wireless-capsule-endoscopy-wce-image-classificat]  
 

For questions or collaborations, feel free to reach out! 🚀  

---

## 📢 Connect with Me  
[![LinkedIn](www.linkedin.com/in/arif-miah-8751bb217)  

 

#MachineLearning #DeepLearning #AI #MedicalImaging #ComputerVision #HealthcareAI  

