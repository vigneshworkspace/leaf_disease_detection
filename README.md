# leaf_disease_detection
# 🍅 Leaf Disease Classifier using DenseNet121 (Multi-label)

This project implements a **multi-label classification model** using **DenseNet121** to detect multiple tomato leaf diseases. The model can identify more than one disease in a single image, enabling effective real-world diagnosis.

---

## 📌 Project Overview

- **Model:** DenseNet121 (transfer learning)
- **Task:** Multi-label image classification
- **Dataset:** [Tomato Leaf Dataset – Kaustubh B (Kaggle)](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
- **Framework:** TensorFlow / Keras
- **File:** `model.h5` (trained model, ~31MB)

---

## 🗃️ Dataset Description

- 📦 **Source:** Kaggle – [Tomato Leaf Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
- 🖼️ **Images:** ~16,000 leaf images across multiple classes
- ⚠️ **Diseases Covered:**
  - Tomato Bacterial Spot
  - Tomato Early Blight
  - Tomato Late Blight
  - Tomato Leaf Mold
  - Tomato Septoria Leaf Spot
  - Tomato Spider Mites
  - Tomato Target Spot
  - Tomato Mosaic Virus
  - Tomato Yellow Leaf Curl Virus
  - Healthy

- 🔀 **Used as multi-label:** In real scenarios, multiple diseases may coexist — this project adapts the dataset accordingly.

---

## 🧠 Model Architecture

- **Base:** DenseNet121 (without top classifier)
- **Top Layers:**
  - Global Average Pooling
  - Dense Layer (sigmoid activation for multi-label)
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy, AUC, Precision, Recall

---

## 🏁 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/vigneshworkspace/leaf-disease-classifier.git
cd leaf-disease-classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Load the Trained Model
```python
from tensorflow.keras.models import load_model
model = load_model("model.h5")
```

### 4. Predict a Sample Image
```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("sample_leaf.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0]

# Define your labels
labels = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 
          'Septoria Spot', 'Spider Mites', 'Target Spot', 
          'Mosaic Virus', 'Yellow Curl Virus', 'Healthy']

# Show results
for i, prob in enumerate(pred):
    if prob > 0.5:
        print(f"{labels[i]}: {prob:.2f}")
```

---

## 📊 Evaluation

| Metric     | Value        |
|------------|--------------|
| Accuracy   | 95.20%       |
| Precision  | 95.23%       |
| Recall     | 95.20%       |
| F1 Score   | 95.18%       |

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       100
           1       0.92      0.96      0.94       100
           2       0.92      0.97      0.94       100
           3       0.94      0.90      0.92       100
           4       0.98      0.92      0.95       100
           5       0.95      0.90      0.92       100
           6       0.92      0.91      0.91       100
           7       0.97      0.99      0.98       100
           8       0.96      1.00      0.98       100
           9       0.98      0.98      0.98       100

    accuracy                           0.95      1000
   macro avg       0.95      0.95      0.95      1000
weighted avg       0.95      0.95      0.95      1000
```

---

## 📁 Project Structure

```
├── model.h5                 # Trained DenseNet121 model
├── predict.py               # Inference script
├── train_model.ipynb        # Training notebook
├── requirements.txt         # Dependencies
├── README.md                # Project overview
└── /dataset                 # (download manually from Kaggle)
```

---

## 🚀 Future Enhancements

- Build a web app with Streamlit / Flask
- Train with synthetic or real-world multi-labeled images
- Add Grad-CAM visualizations for explainability

---

## 🧑‍🔬 Author

**Vignesh**  
🔗 [LinkedIn](https://www.linkedin.com/in/vignesh-sist/) • 📸 [Instagram](https://instagram.com/itsvignesh_43)

---

## 📜 License

This project is open-source under the **MIT License**.

![sample_images](https://github.com/user-attachments/assets/956b3504-9f1f-412e-b465-dcfbbb8bbfdb)
