// Add an environment variable for the API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const handleDetection = async () => {
  if (!file) {
    alert('Please upload a fingerprint image first');
    return;
  }

  setIsLoading(true);
  setResult(null);

  try {
    const formData = new FormData();
    formData.append('image', file);

    const response = await axios.post(`${API_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    console.log("Response from backend:", response.data);
    setResult(response.data);
  } catch (error) {
    console.error('Error during prediction:', error);
    if (error.response) {
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
      console.error('Response headers:', error.response.headers);
    } else if (error.request) {
      console.error('No response received:', error.request);
    } else {
      console.error('Error:', error.message);
    }
    alert('An error occurred during prediction. Please try again.');
  } finally {
    setIsLoading(false);
  }
};

{result && (
  <div className="mt-8 p-6 bg-green-50 rounded-lg">
    <h3 className="text-2xl font-bold text-green-800">Blood Group: {result.blood_group}</h3>
    <p className="text-green-600 mt-2">Confidence: {result.confidence}</p>
    <p className="text-green-600 mt-2">Analysis completed successfully</p>
  </div>
)} 