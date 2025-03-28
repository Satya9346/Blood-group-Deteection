import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { PythonShell } from 'python-shell';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

const app = express();
const port = 3001;

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

app.use(cors());
app.use(express.json());

// Endpoint to handle fingerprint image upload and prediction
app.post('/api/predict', upload.single('fingerprint'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const imagePath = req.file.path;
  
  const options = {
    mode: 'text',
    pythonPath: 'python3',
    scriptPath: __dirname,
    args: [imagePath]
  };

  PythonShell.run('blood_group_predictor.py', options, function (err, results) {
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    
    try {
      const result = JSON.parse(results[0]);
      return res.json(result);
    } catch (e) {
      return res.status(500).json({ error: 'Failed to parse prediction result' });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});