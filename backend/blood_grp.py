import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import numpy as np
from PIL import Image
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import tensorflow as tf
from pathlib import Path
import sys
import traceback
from werkzeug.utils import secure_filename

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# Get port from environment variable
port = int(os.environ.get("PORT", 10000))
print(f"=== Environment Configuration ===")
print(f"PORT env var: {os.environ.get('PORT')}")
print(f"Configured port: {port}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Update CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",  # Vite default port
            "http://localhost:5174",  # Alternative Vite port
            "http://localhost:3000",   # Just in case
            "https://blood-group-deteection.onrender.com",
            "https://blood-group-detection-frontend.onrender.com"  # Frontend Render URL
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Constants
IMG_HEIGHT = 64
IMG_WIDTH = 64
MODEL_PATH = 'model/blood_group_mode.h5'

print(f"Using model path: {MODEL_PATH}")  # Debug print

# Add this after MODEL_PATH definition
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model file not found at {MODEL_PATH}")
    print(f"Current directory contents: {os.listdir('.')}")

# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BloodGroupPredictor:
    def __init__(self, model_path=MODEL_PATH):
        """Initialize the predictor with model path"""
        print(f"Loading model from: {model_path}")
        self.classes = ['A+', 'A-', 'B+', 'AB-', 'AB+', 'B-', 'O+', 'O-']
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            print("Model file exists, attempting to load...")
            
            try:
                # First attempt: normal loading
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            except ValueError as e:
                print(f"First load attempt failed: {str(e)}")
                print("Attempting to reconstruct model from scratch...")
                
                # Recreate the exact model architecture from training
                self.model = tf.keras.models.Sequential([
                    # Input layer
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    
                    # First block
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Dropout(0.25),
                    
                    # Second block
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Dropout(0.25),
                    
                    # Third block
                    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Dropout(0.25),
                    
                    # Dense layers
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(8, activation='softmax')  # 8 classes for blood groups
                ])
                
                # Load weights
                print("Loading weights into reconstructed model...")
                self.model.load_weights(model_path)
            
            # Compile model with same configuration as training
            print("Compiling model...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Model loaded and compiled successfully")
            
            # Warm up the model with a dummy prediction
            print("Performing model warm-up...")
            dummy_input = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3))
            self.model.predict(dummy_input, verbose=0)
            print("Model warm-up complete")
        except Exception as e:
            print(f"Detailed error loading model: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            self.model = None

    def preprocess_image(self, img):
        """Preprocess the input image for prediction"""
        try:
            print(f"Original image mode: {img.mode}")
            print(f"Original image size: {img.size}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                print(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
            
            # Apply basic image enhancement
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)  # Increase contrast slightly
            
            # Resize to match training dimensions
            print(f"Resizing image to ({IMG_HEIGHT}, {IMG_WIDTH})")
            img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
            
            # Convert to array and normalize
            img_array = np.array(img)
            
            # Apply additional preprocessing
            img_array = img_array.astype('float32')
            
            # Normalize to [0,1]
            img_array = img_array / 255.0
            
            # Apply standardization
            mean = np.mean(img_array)
            std = np.std(img_array)
            img_array = (img_array - mean) / (std + 1e-7)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            print(f"Preprocessed image shape: {img_array.shape}")
            print(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            return img_array
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise

    def predict_blood_group(self, image_path=None, image_data=None):
        """Predict blood group from image path or image data"""
        try:
            if image_path:
                print(f"Loading image from path: {image_path}")
                if not os.path.exists(image_path):
                    return {"error": "Image file not found"}
                img = Image.open(image_path)
            elif image_data:
                print("Loading image from uploaded data")
                img = Image.open(io.BytesIO(image_data))
            else:
                return {"error": "No image provided"}

            # Preprocess image
            print("Preprocessing image...")
            img_array = self.preprocess_image(img)
            
            if self.model is None:
                print("ERROR: Model not loaded properly")
                return {"error": "Model not loaded properly"}
            
            # Make prediction using the model
            print("Making prediction...")
            predictions = self.model.predict(img_array)
            
            # Get top 2 predictions
            top_2_indices = np.argsort(predictions[0])[-2:][::-1]
            top_2_confidences = predictions[0][top_2_indices] * 100
            
            predicted_class_index = top_2_indices[0]
            confidence = float(top_2_confidences[0])
            second_confidence = float(top_2_confidences[1])
            
            # If confidence is too low or top 2 predictions are too close, return uncertain
            CONFIDENCE_THRESHOLD = 60.0  # 60% confidence threshold
            CONFIDENCE_DIFF_THRESHOLD = 20.0  # 20% difference threshold
            
            if confidence < CONFIDENCE_THRESHOLD or (confidence - second_confidence) < CONFIDENCE_DIFF_THRESHOLD:
                return {
                    'blood_group': 'Uncertain',
                    'confidence': f"{confidence:.2f}%",
                    'second_prediction': {
                        'blood_group': self.classes[top_2_indices[1]],
                        'confidence': f"{second_confidence:.2f}%"
                    },
                    'predictions': predictions[0].tolist()
                }
            
            print(f"Raw predictions: {predictions[0]}")
            print(f"Predicted class index: {predicted_class_index}")
            print(f"Blood group: {self.classes[predicted_class_index]}")
            print(f"Confidence: {confidence:.2f}%")
            
            return {
                'blood_group': self.classes[predicted_class_index],
                'confidence': f"{confidence:.2f}%",
                'second_prediction': {
                    'blood_group': self.classes[top_2_indices[1]],
                    'confidence': f"{second_confidence:.2f}%"
                },
                'predictions': predictions[0].tolist()
            }

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {"error": str(e)}

def test_dataset(dataset_path):
    """Test predictions on all images in the dataset"""
    predictor = BloodGroupPredictor()
    results = []
    total_images = 0
    correct_predictions = 0

    try:
        # Get all blood group folders
        blood_groups = [d for d in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, d))]
        
        summary = {
            "total_images": 0,
            "correct_predictions": 0,
            "accuracy": 0,
            "results": []
        }

        for blood_group in blood_groups:
            group_path = os.path.join(dataset_path, blood_group)
            
            # Get all BMP images in this blood group folder
            images = [f for f in os.listdir(group_path) 
                     if f.lower().endswith('.bmp')]
            
            for image in images:
                total_images += 1
                image_path = os.path.join(group_path, image)
                result = predictor.predict_blood_group(image_path=image_path)
                
                if "error" not in result:
                    predicted_group = result['blood_group']
                    confidence = result['confidence']
                    is_correct = predicted_group == blood_group
                    if is_correct:
                        correct_predictions += 1
                    
                    results.append({
                        'file': image,
                        'actual': blood_group,
                        'predicted': predicted_group,
                        'confidence': confidence,
                        'correct': is_correct
                    })

        # Calculate accuracy
        accuracy = (correct_predictions / total_images * 100) if total_images > 0 else 0
        
        summary["total_images"] = total_images
        summary["correct_predictions"] = correct_predictions
        summary["accuracy"] = accuracy
        summary["results"] = results
        
        return summary
    except Exception as e:
        return {"error": str(e)}

# Web Routes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug("=== Starting prediction request ===")
        logger.debug(f"Request headers: {dict(request.headers)}")
        
        # Check if file exists in request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        logger.debug(f"Received file: {file.filename}")
        
        # Check if file is empty
        if file.filename == '':
            logger.error("Empty filename received")
            return jsonify({"error": "No selected file"}), 400
        
        try:
            # Read image file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            logger.debug(f"Image opened successfully. Format: {image.format}, Size: {image.size}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.debug("Converted image to RGB")
            
            # Resize image to 64x64 as expected by the model
            image = image.resize((64, 64))
            logger.debug("Image resized to 64x64")
            
            # Convert to numpy array
            img_array = np.array(image)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)
            logger.debug(f"Image array shape: {img_array.shape}")
            
            # Load model
            model_path = MODEL_PATH
            logger.debug(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return jsonify({"error": f"Model file not found"}), 500
            
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.debug("Model loaded successfully")
            
            # Make prediction
            prediction = model.predict(img_array)
            logger.debug(f"Raw prediction: {prediction}")
            
            # Process prediction result
            blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            blood_group = blood_groups[predicted_class]
            
            logger.debug(f"Predicted blood group: {blood_group}, Confidence: {confidence}")
            
            return jsonify({
                "blood_group": blood_group,
                "confidence": confidence,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Error processing image",
                "details": str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500

@app.route('/test-dataset', methods=['POST'])
def test_dataset_endpoint():
    """Handle dataset testing request"""
    try:
        data = request.get_json()
        if not data or 'dataset_path' not in data:
            return jsonify({"error": "Dataset path not provided"}), 400
            
        dataset_path = data['dataset_path']
        if not os.path.exists(dataset_path):
            return jsonify({"error": "Dataset folder not found"}), 404
            
        results = test_dataset(dataset_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        is_valid, error_message = is_valid_bmp(file)
        if not is_valid:
            logger.error(f"Invalid file type: {error_message}")
            return jsonify({
                "error": "Invalid file type",
                "message": error_message,
                "allowed_extensions": list(ALLOWED_EXTENSIONS)
            }), 415  # 415 Unsupported Media Type
        
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file temporarily
            file.save(file_path)
            logger.info(f"File saved temporarily at: {file_path}")
            
            # Initialize the predictor
            try:
                predictor = BloodGroupPredictor()
            except Exception as model_error:
                logger.error(f"Error initializing predictor: {str(model_error)}")
                logger.error(traceback.format_exc())
                # Clean up file in case of error
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up temporary file: {file_path}")
                    except:
                        pass
                return jsonify({"error": f"Model initialization error: {str(model_error)}"}), 500
            
            # Get prediction
            try:
                result = predictor.predict_blood_group(image_path=file_path)
            except Exception as pred_error:
                logger.error(f"Error during prediction: {str(pred_error)}")
                logger.error(traceback.format_exc())
                # Clean up file in case of error
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up temporary file: {file_path}")
                    except:
                        pass
                return jsonify({"error": f"Prediction error: {str(pred_error)}"}), 500
            
            # Delete the input file after processing
            try:
                os.remove(file_path)
                logger.info(f"Temporary file deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            if "error" in result:
                logger.error(f"Error in prediction result: {result['error']}")
                return jsonify(result), 500
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            # Clean up file in case of error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
                except:
                    pass
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def health_check():
    current_port = int(os.environ.get("PORT", 10000))
    return jsonify({
        "status": "healthy",
        "message": "Blood Group Detection API is running",
        "port": current_port,
        "environment": {
            "PYTHON_VERSION": os.environ.get("PYTHON_VERSION"),
            "PORT": os.environ.get("PORT"),
            "PWD": os.getcwd(),
            "PATH": os.environ.get("PATH")
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 