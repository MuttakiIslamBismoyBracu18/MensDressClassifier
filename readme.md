# 👔 Men's Dress Classifier

This repository contains a **CNN-based classification system** that detects whether a person is wearing **Formal** or **Informal** clothing. The system can classify both **individual images** and **videos** using a pre-trained deep learning model.

---

## 🚀 Features
- ✅ **Image Classification**: Upload an image and determine whether the dress is **Formal** or **Informal**.
- ✅ **Video Classification**: Extracts frames from a video and determines the **dominant clothing type** using majority voting.
- ✅ **CNN Model**: Trained on a custom dataset using **TensorFlow**.
- ✅ **High Accuracy**: Uses **data augmentation, dropout, and fine-tuning** for precision.
- ✅ **Error Handling**: Detects missing files, corrupt images, and invalid inputs.

---

## 📂 Folder Structure
📁 Men'sDressClassifier │── 📁 Model_Training # Model training scripts and dataset │ │── train.py # CNN model training script │ │── dataset/ # Train, validation, and test data (Formal & Informal) │ │── formal_informal_classifier.h5 # Trained model file │ │── 📁 System # Inference and classification system │ │── image.py # Script for image classification │ │── video.py # Script for video classification │ │── 📁 Samples # Sample images/videos for testing │── README.md # Project documentation


---

## 📊 Model Training
The CNN model is trained using **TensorFlow & Keras**. The dataset consists of **Formal and Informal** clothing images in **train, valid, and test** folders.

### 🛠 **Training Steps**
1. Load dataset using **ImageDataGenerator**.
2. Apply **data augmentation** for better generalization.
3. Train a **CNN model** with dropout to prevent overfitting.
4. Save the trained model as **`formal_informal_classifier.h5`**.

🔹 **Training Script:** [`train.py`](Model_Training/train.py)

---

## 🖼️ Image Classification
Classifies an **image** as **Formal or Informal** using the trained model.

### ✅ **Usage**
```bash
python image.py

or modify image.py to use a specific image:

python
Copy
Edit
predict_image("Samples/formal_sample.jpg")
🔹 Script: image.py

🎥 Video Classification
Analyzes a video and determines the dominant dress type using frame-wise classification.

✅ Usage
bash
Copy
Edit
python video.py
or modify video.py to use a specific video:

python
Copy
Edit
process_video("Samples/test_video.mp4")
🔹 Script: video.py

📦 Dependencies
Make sure to install the required libraries:

bash
Copy
Edit
pip install tensorflow opencv-python numpy
🎯 Future Improvements
🚀 Improve Accuracy: Use a larger dataset and train with ResNet50 or EfficientNet.
📱 Deploy as a Web App: Convert this into a Flask/Streamlit app.
⚡ Optimize Model: Convert to TensorFlow Lite for mobile use.
🤝 Contributing
Feel free to fork this repository and contribute to improving the accuracy and usability of the system! If you have any suggestions, open an issue or create a pull request.

📜 License
This project is MIT Licensed. You are free to use, modify, and distribute it.

📬 Contact
📧 Email: bismoym.mail.gvsu.edu