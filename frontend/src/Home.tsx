// Add an environment variable for the API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Update your axios calls
const response = await axios.post(`${API_URL}/predict`, formData, {
    headers: {
        'Content-Type': 'multipart/form-data'
    }
}); 