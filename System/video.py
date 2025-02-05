import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
import os

model_path = r"E:\PycharmProjects\MensDressClassifier\Model_Training\formal_informal_classifier.h5"

if os.path.exists(model_path):
    print("✅ Model file found. Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully.")
else:
    print("❌ Model file not found!")
    exit()

class_labels = ["Informal", "Formal"] 

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0) 
    img = img.astype("float32") / 255.0  
    return img

def classify_frame(frame):
    img = preprocess_frame(frame)
    predictions = model.predict(img)  

    if predictions is None or len(predictions) == 0:
        print("⚠️ Model did not return any predictions.")
        return None

    label_idx = int(predictions[0][0] >= 0.31) 
    #print("Label Index : ",label_idx) 
    return class_labels[label_idx]

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_interval = 10  
    frame_count = 0
    predictions = []

    if not cap.isOpened():
        print("⚠️ Error: Cannot open video file!")
        return "Error: Cannot open video"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ No more frames to read at frame {frame_count}.")
            break

        if frame_count % frame_interval == 0:
            label = classify_frame(frame)
            if label: 
                predictions.append(label)
                print(f"✅ Frame {frame_count}: {label}")
            else:
                print(f"⚠️ No classification result for frame {frame_count}")

        frame_count += 1

    cap.release()

    if not predictions:
        print("❌ Error: No predictions made. Check video format and model.")
        return "Error: No predictions"

    most_common_label = Counter(predictions).most_common(1)[0][0]
    return most_common_label

video_file = r"E:\PycharmProjects\MensDressClassifier\2.mp4"  
final_prediction = process_video(video_file)
print(f"🎯 Final Classification: {final_prediction}")