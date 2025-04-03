import os
import numpy as np
from PIL import Image
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
IMG_HEIGHT = 64
IMG_WIDTH = 64
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'blood_group_mode.h5')

print(f"Using model path: {MODEL_PATH}")  # Debug print

class BloodGroupPredictor:
    def __init__(self, model_path=MODEL_PATH):
        """Initialize the predictor with model path"""
        print(f"Loading model from: {model_path}")
        self.classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            print("Model file exists, attempting to load...")
            
            try:
                # First attempt: normal loading
                self.model = tf.keras.models.load_model(model_path, compile=False)
            except ValueError as e:
                if 'batch_shape' in str(e):
                    print("Attempting alternative model loading method...")
                    # Load model architecture and weights separately
                    with tf.keras.utils.custom_object_scope({'BatchNormalization': tf.keras.layers.BatchNormalization}):
                        self.model = tf.keras.models.load_model(
                            model_path,
                            compile=False,
                            options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                        )
                else:
                    raise e
            
            # Compile model
            self.model.compile(
                optimizer='adam',
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
            # Debug print original image properties
            print(f"Original image mode: {img.mode}")
            print(f"Original image size: {img.size}")
            
            # Convert to RGB if image is in a different mode
            if img.mode != 'RGB':
                print(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
            
            # Resize image to required dimensions
            print(f"Resizing image from {img.size} to ({IMG_HEIGHT}, {IMG_WIDTH})")
            img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.Resampling.LANCZOS)
            
            # Convert to array and normalize
            img_array = np.array(img)
            print(f"Image array shape before normalization: {img_array.shape}")
            print(f"Image array value range before normalization: [{img_array.min()}, {img_array.max()}]")
            
            # Normalize exactly like in training
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            print(f"Final preprocessed image shape: {img_array.shape}")
            print(f"Final value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            return img_array
        except Exception as e:
            print(f"Detailed preprocessing error: {str(e)}")
            raise Exception(f"Error preprocessing image: {str(e)}")

    def predict_blood_group(self, image_path=None, image_data=None):
        """Predict blood group from image path or image data"""
        try:
            if image_path:
                print(f"Loading image from path: {image_path}")  # Added debug print
                if not os.path.exists(image_path):
                    return {"error": "Image file not found"}
                img = Image.open(image_path)
            elif image_data:
                print("Loading image from uploaded data")  # Added debug print
                img = Image.open(io.BytesIO(image_data))
            else:
                return {"error": "No image provided"}

            # Preprocess image
            print("Preprocessing image...")  # Added debug print
            img_array = self.preprocess_image(img)
            
            if self.model is None:
                print("ERROR: Model not loaded properly")  # Added debug print
                return {"error": "Model not loaded properly"}
            
            # Make prediction using the model
            print("Making prediction...")  # Added debug print
            predictions = self.model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index] * 100)
            
            print(f"Raw predictions: {predictions[0]}")
            print(f"Predicted class index: {predicted_class_index}")
            print(f"Blood group: {self.classes[predicted_class_index]}")
            print(f"Confidence: {confidence:.2f}%")
            
            return {
                'blood_group': self.classes[predicted_class_index],
                'confidence': f"{confidence:.2f}%",
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
    """Handle image prediction request"""
    try:
        print("=== Starting new prediction request ===")
        if 'image' not in request.files:
            print("No image file in request")
            print("Request files:", request.files)
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            print("Empty filename")
            return jsonify({"error": "No selected file"}), 400
            
        # Print file information
        print(f"Received file: {file.filename}")
        print(f"File content type: {file.content_type}")
        print(f"File headers: {dict(file.headers)}")
            
        # Verify file is a BMP
        if not file.filename.lower().endswith('.bmp'):
            print(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Only BMP files are supported"}), 400
            
        # Read the image file
        image_data = file.read()
        print(f"Read {len(image_data)} bytes of image data")
        
        # Try to open the image to verify it's valid
        try:
            test_img = Image.open(io.BytesIO(image_data))
            print(f"Successfully opened image: format={test_img.format}, size={test_img.size}, mode={test_img.mode}")
        except Exception as img_error:
            print(f"Failed to open image: {str(img_error)}")
            return jsonify({"error": f"Invalid image file: {str(img_error)}"}), 400
        
        # Create predictor and get prediction
        predictor = BloodGroupPredictor()
        
        # Check if model loaded correctly
        if predictor.model is None:
            print("Model failed to load")
            return jsonify({"error": "Model failed to load"}), 500
            
        result = predictor.predict_blood_group(image_data=image_data)
        print(f"Prediction result: {result}")
        
        if "error" in result:
            print(f"Error in prediction: {result['error']}")
            return jsonify(result), 500
            
        print("=== Prediction completed successfully ===")
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"Prediction error: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    # If running as script, start Flask server
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True) 