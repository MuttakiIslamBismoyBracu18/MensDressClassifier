from tensorflow.keras.models import load_model
import cv2
import numpy as np


model_path = r"E:\PycharmProjects\MensDressClassifier\Model_Training\formal_informal_classifier.h5"
model = load_model(model_path)  

def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Unable to load image. Check file path!")
        return

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0  

    prediction = model.predict(img)[0][0] 
    label = "Formal" if prediction >= 0.31 else "Informal"
    print(f"🎯 Predicted {prediction:.2f} Class: {label}")

predict_image(r"E:\PycharmProjects\MensDressClassifier\3.jpg")
