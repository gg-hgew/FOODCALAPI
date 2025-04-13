from flask import Flask, request, jsonify 
import torch
import cv2
import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
from google.generativeai import GenerativeModel, configure
from werkzeug.utils import secure_filename

# ✅ Initialize App
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Gemini API Configuration
GEMINI_API_KEY = "AIzaSyCJIWi78tzKE_dDSK9FKazGxGBn5I2gd2E"
configure(api_key=GEMINI_API_KEY)
gemini_model = GenerativeModel(model_name="gemini-1.5-pro")

# ✅ Load YOLOv8
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# ✅ Food Tables
portion_sizes = {
    "pizza": 120, "burger": 150, "cake": 80, "fries": 140, "pasta": 200,
    "rice": 180, "sushi": 130, "steak": 250, "chicken": 200, "fish": 220,
    "apple": 150, "banana": 120, "grapes": 100, "bread": 50, "cheese": 30,
    "potato": 180, "carrot": 100, "tomato": 80, "milk": 250, "soda": 330
}
calorie_table = {
    "pizza": 266, "burger": 295, "cake": 350, "fries": 312, "pasta": 158,
    "rice": 130, "sushi": 150, "steak": 271, "chicken": 239, "fish": 206,
    "apple": 52, "banana": 89, "grapes": 69, "bread": 265, "cheese": 402,
    "potato": 77, "carrot": 41, "tomato": 18, "milk": 42, "soda": 150
}

# ✅ AI Functions
def gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the AI Food Calorie Estimator API! Use /analyze to upload an image."})

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Empty response to avoid 404 for favicon.ico

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Check if the request contains an image
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded!'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # Read and process the image
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    results = model.predict(source=path, conf=0.5)
    
    output_data = []

    # Process detection results
    for result in results:
        for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
            class_name = model.names[int(cls_id)]
            food_name = class_name.lower()
            portion_weight = portion_sizes.get(food_name, 100)
            estimated_calories = (calorie_table.get(food_name, 0) / 100) * portion_weight

            entry = {
                'food_name': food_name,
                'estimated_calories': round(estimated_calories, 2),
                'workout_plan': gemini_response(f"Create a workout plan to burn {estimated_calories} kcal."),
                'dietary_advice': gemini_response(f"Provide dietary recommendations for {food_name} with {estimated_calories} kcal."),
                'meal_plan': gemini_response(f"Generate a balanced meal plan including {food_name} with {estimated_calories} kcal.")
            }
            output_data.append(entry)

    # Return the results
    return jsonify({'results': output_data})

if __name__ == '__main__':
    app.run(debug=True)
