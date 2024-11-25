from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
from PIL import Image, ImageDraw
import torchvision.transforms as T
import pickle

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


model_path = r'C:\Users\tw93\OneDrive\Desktop\masters_project\trained_model1.keras'
classification_model = load_model(model_path)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


calories = {
    'apple': 52, 'banana': 96, 'bhatura': 300, 'brownie': 466, 'burger': 295,
    'chapati': 68, 'chole': 164, 'dhokla': 160, 'gulabjamun': 175,
    'kababs': 197, 'mango': 60, 'meduvada': 135, 'modak': 108,
    'pizza': 266, 'pomogranate': 83, 'rice': 130, 'samosa': 262, 'strawberry': 33,
    'vadapav': 300, 'watermelon': 30
}


detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
detection_model.eval()


transform = T.Compose([
    T.ToTensor(),
])

def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    else:
        raise ValueError(f"Image not found or unable to read: {image_path}")

def detect_objects(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        detections = detection_model(image_tensor)[0]
    return detections, image

def filter_detections(detections, score_threshold=0.5, iou_threshold=0.3):
    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']
    keep = nms(boxes, scores, iou_threshold)
    filtered_boxes = boxes[keep][scores[keep] > score_threshold]
    filtered_labels = labels[keep][scores[keep] > score_threshold]
    return filtered_boxes, filtered_labels

def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle(box.tolist(), outline='red', width=3)
        draw.text((box[0], box[1]), str(label), fill='red')
    return image

def classify_objects(image_path, boxes):
    detected_labels = []
    for box in boxes:
        crop = cv2.imread(image_path)[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        if crop.size > 0:
            crop = cv2.resize(crop, (128, 128))
            crop = crop.astype('float32') / 255.0
            crop = np.expand_dims(crop, axis=0)
            prediction = classification_model.predict(crop)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
            
            # Ensure the predicted label is a food item
            if predicted_class_label in calories:
                detected_labels.append(predicted_class_label)
    
    return detected_labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            
            detections, image = detect_objects(filepath)
            filtered_boxes, filtered_labels = filter_detections(detections)

            
            detected_labels = classify_objects(filepath, filtered_boxes)

            #
            total_calories = sum(calories.get(label, 0) for label in detected_labels)

            
            image_with_boxes = draw_boxes(image, filtered_boxes, detected_labels)
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            image_with_boxes.save(result_image_path)

            return jsonify({
                'detected_labels': detected_labels,
                'total_calories': total_calories,
                'image': '/' + result_image_path
            })

    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'error': 'Unknown error'})

if __name__ == '__main__':
    app.run(debug=True)
