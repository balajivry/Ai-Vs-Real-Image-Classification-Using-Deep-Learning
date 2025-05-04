from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['MODEL_PATH'] = 'best_model.keras'

# Load the model
model = tf.keras.models.load_model(app.config['MODEL_PATH'])

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))  # Match model's expected input size
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)
            
            # Handle binary or multi-class output
            if prediction.shape[1] == 1:  # Binary classification
                result = "Real Image" if prediction[0][0] > 0.5 else "AI-Generated Image"
                confidence = round((prediction[0][0] if prediction[0][0] > 0.5 else (1 - prediction[0][0])) * 100)
            else:  # Multi-class classification
                result = "Real Image" if np.argmax(prediction) == 1 else "AI-Generated Image"
                confidence = round(np.max(prediction) * 100, 2)
            
            # Relative path for HTML template
            image_path = os.path.join('uploads', filename)
            
            return render_template('result.html',
                                prediction=result,
                                confidence=confidence,
                                image_file=filename)
            
        except Exception as e:
            print(f"Error: {e}")
            return redirect(url_for('home'))
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)