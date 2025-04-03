import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
port = int(os.environ.get("PORT", 10000))

CORS(app, resources={
    r"/*": {
        "origins": "*"  # Allow all origins in development
    }
})

# Add a basic route for health checks
@app.route('/')
def health_check():
    return jsonify({"status": "healthy"}), 200

# ... rest of your code ...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port) 