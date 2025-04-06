import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import magic  # for MIME type detection
import tensorflow as tf
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
port = int(os.environ.get("PORT", 10000))

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'bmp'}
ALLOWED_MIMETYPES = {'image/bmp', 'image/x-bmp', 'image/x-ms-bmp'}

# Set the model path directly from GitHub
MODEL_PATH = 'backend/model/blood_group_mode.h5'

logger.info(f"Using model file: {MODEL_PATH}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app, resources={
    r"/*": {
        "origins": "*"  # Allow all origins in development
    }
})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_bmp(file):
    # Check file extension
    if not allowed_file(file.filename):
        return False, "Invalid file extension"
    
    # Check MIME type
    try:
        mime = magic.from_buffer(file.read(2048), mime=True)
        file.seek(0)  # Reset file pointer
        if mime not in ALLOWED_MIMETYPES:
            return False, f"Invalid file format. Expected BMP, got {mime}"
    except Exception as e:
        return False, f"Error checking file format: {str(e)}"
    
    # Verify it's a valid BMP using PIL
    try:
        img = Image.open(file)
        file.seek(0)  # Reset file pointer
        if img.format != 'BMP':
            return False, "File is not a valid BMP image"
    except Exception as e:
        return False, f"Error validating BMP: {str(e)}"
    
    return True, None

class BloodGroupPredictor:
    def __init__(self, model_path=MODEL_PATH):
        """Initialize the predictor with model path"""
        logger.info(f"Initializing BloodGroupPredictor with model path: {model_path}")
        self.classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        try:
            if model_path is None:
                error_msg = "No model file found. Please ensure the model file exists in the model directory."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            if not os.path.exists(model_path):
                error_msg = f"Model file not found at: {model_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info("Model file exists, attempting to load...")
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("Model loaded successfully")
                
                # Compile the model
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("Model compiled successfully")
                
            except Exception as model_error:
                logger.error(f"Error loading/compiling model: {str(model_error)}")
                logger.error(traceback.format_exc())
                raise
            
        except Exception as e:
            logger.error(f"Error in BloodGroupPredictor initialization: {str(e)}")
            logger.error(traceback.format_exc())
            self.model = None
            raise

    def preprocess_image(self, img):
        """Preprocess the input image for prediction"""
        try:
            logger.info(f"Original image mode: {img.mode}")
            logger.info(f"Original image size: {img.size}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                logger.info(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
            
            # Resize to match training dimensions
            logger.info(f"Resizing image to (64, 64)")
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Convert to array and normalize
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            logger.info(f"Preprocessed image shape: {img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict_blood_group(self, image_path=None, image_data=None):
        """Predict blood group from image path or image data"""
        try:
            if self.model is None:
                error_msg = "Model not loaded properly"
                logger.error(error_msg)
                return {"error": error_msg}
            
            if image_path:
                logger.info(f"Loading image from path: {image_path}")
                if not os.path.exists(image_path):
                    error_msg = "Image file not found"
                    logger.error(error_msg)
                    return {"error": error_msg}
                img = Image.open(image_path)
            elif image_data:
                logger.info("Loading image from uploaded data")
                img = Image.open(io.BytesIO(image_data))
            else:
                error_msg = "No image provided"
                logger.error(error_msg)
                return {"error": error_msg}

            # Preprocess image
            logger.info("Preprocessing image...")
            img_array = self.preprocess_image(img)
            
            # Make prediction
            logger.info("Making prediction...")
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top 2 predictions
            top_2_indices = np.argsort(predictions[0])[-2:][::-1]
            top_2_confidences = predictions[0][top_2_indices] * 100
            
            predicted_class_index = top_2_indices[0]
            confidence = float(top_2_confidences[0])
            second_confidence = float(top_2_confidences[1])
            
            logger.info(f"Predicted blood group: {self.classes[predicted_class_index]}")
            logger.info(f"Confidence: {confidence:.2f}%")
            
            return {
                'blood_group': self.classes[predicted_class_index],
                'confidence': f"{confidence:.2f}%",
                'second_prediction': {
                    'blood_group': self.classes[top_2_indices[1]],
                    'confidence': f"{second_confidence:.2f}%"
                }
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

# Add a basic route for health checks
@app.route('/')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        is_valid, error_message = is_valid_bmp(file)
        if not is_valid:
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
            predictor = BloodGroupPredictor()
            
            # Get prediction
            result = predictor.predict_blood_group(image_path=file_path)
            
            # Delete the input file after processing
            try:
                os.remove(file_path)
                logger.info(f"Temporary file deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            if "error" in result:
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port) 