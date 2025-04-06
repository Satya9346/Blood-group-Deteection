import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import magic  # for MIME type detection

app = Flask(__name__)
port = int(os.environ.get("PORT", 10000))

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'bmp'}
ALLOWED_MIMETYPES = {'image/bmp', 'image/x-bmp', 'image/x-ms-bmp'}

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

# Add a basic route for health checks
@app.route('/')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
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
        
        # Initialize the predictor
        predictor = BloodGroupPredictor()
        
        # Get prediction
        result = predictor.predict_blood_group(image_path=file_path)
        
        # Delete the input file after processing
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")
        
        return jsonify(result), 200
        
    except Exception as e:
        # Clean up file in case of error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({"error": str(e)}), 500

# ... rest of your code ...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port) 