# BloodPrint - Blood Group Detection via Fingerprint Analysis

This project uses AI to predict blood groups from fingerprint images. It consists of a React frontend and a Python backend.

## Project Structure

```
bloodprint/
├── backend/                # Python backend
│   ├── blood_group_predictor.py  # Python prediction script
│   ├── model/              # Directory for the ML model
│   ├── package.json        # Node.js dependencies for the server
│   ├── requirements.txt    # Python dependencies
│   └── server.js           # Express server
├── frontend/               # React frontend
│   ├── public/             # Static files
│   ├── src/                # React source code
│   ├── index.html          # HTML template
│   ├── package.json        # Frontend dependencies
│   ├── vite.config.ts      # Vite configuration
│   └── ...                 # Other configuration files
└── uploads/                # Uploaded fingerprint images
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install Node.js dependencies:
   ```
   npm install
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a model directory (if you have a trained model):
   ```
   mkdir -p model
   ```

5. Place your trained model in the model directory (if available).

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Running the Application

### Start the Backend Server

1. From the backend directory:
   ```
   npm start
   ```
   This will start the Express server on port 3001.

### Start the Frontend Development Server

1. From the frontend directory:
   ```
   npm run dev
   ```
   This will start the Vite development server.

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:5173).

## Usage

1. Upload a fingerprint image by dragging and dropping or using the file browser.
2. Click "Detect Blood Group" to send the image to the backend for analysis.
3. View the predicted blood group and confidence level.

## Notes

- The current implementation uses simulated results if no actual model is available.
- For production use, you would need to train and include a real TensorFlow model.

## Deployment Instructions

### Backend (Render.com)
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn backend.blood_grp:app`
   - Python Version: 3.9.13

### Frontend (Render.com)
1. Create a new Static Site on Render
2. Connect your GitHub repository
3. Use these settings:
   - Build Command: `npm install && npm run build`
   - Publish Directory: `dist`
   - Environment Variables: Add `VITE_API_URL=https://your-backend-url.onrender.com`

### Environment Variables
- Set `PYTHON_VERSION=3.9.13` in your backend service
- Set `VITE_API_URL` in your frontend service