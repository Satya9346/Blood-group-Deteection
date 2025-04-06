const handleDetection = async () => {
  if (!file) {
    alert('Please upload a fingerprint image first');
    return;
  }

  setIsLoading(true);
  setError(null);

  try {
    const formData = new FormData();
    formData.append('image', file);

    console.log('Sending request to:', `${import.meta.env.VITE_API_URL}/predict`);
    console.log('File being sent:', file);

    const response = await axios.post(`${import.meta.env.VITE_API_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      withCredentials: true
    });

    console.log('Response:', response.data);
    setResult(response.data);
  } catch (error) {
    console.error('Error during prediction:', error);
    setError('An error occurred during prediction. Please try again');
  } finally {
    setIsLoading(false);
  }
}; 