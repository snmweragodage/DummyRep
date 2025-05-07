from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, Upload
import tensorflow as tf
import os
import numpy as np
import cv2
import config

app = Flask(__name__)
app.config.from_object('config')
CORS(app)
db.init_app(app)

# Load AI model
model = tf.keras.models.load_model("skincancermodel.h5")

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image_path = os.path.join(config.UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0

    # AI Prediction
    prediction = model.predict(img)
    classes = ['Benign', 'Non-Melanoma', 'Melanoma']
    result = classes[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    # Save to DB
    upload = Upload(image_path=image_path, prediction=result, risk_score=confidence)
    db.session.add(upload)
    db.session.commit()

    return jsonify({"prediction": result, "confidence": f"{confidence}%"})

if __name__ == '__main__':
    app.run(debug=True)
