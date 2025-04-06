import React, { useState, useRef } from 'react';
import { Upload, Loader } from 'lucide-react';
import axios from 'axios';

function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{ blood_group: string; confidence: string } | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    // Check file extension
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension !== 'bmp') {
      setError(`Invalid file type. Only .bmp files are allowed. Received: .${extension}`);
      return false;
    }

    // Check MIME type
    const validMimeTypes = ['image/bmp', 'image/x-bmp', 'image/x-ms-bmp'];
    if (!validMimeTypes.includes(file.type.toLowerCase())) {
      setError(`Invalid file format. Only .bmp files are allowed.`);
      return false;
    }

    setError(null);
    return true;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (validateFile(droppedFile)) {
        setFile(droppedFile);
      } else {
        // Clear the file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
      } else {
        // Reset the file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleDetection = async () => {
    if (!file) {
      setError('Please upload a fingerprint image first');
      return;
    }

    if (!validateFile(file)) {
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:10000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResult(response.data);
    } catch (error: any) {
      console.error('Error during prediction:', error);
      setError(error.response?.data?.message || 'An error occurred during prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Blood Group Detection via Fingerprint Analysis
        </h1>
        <p className="text-lg text-gray-600">
          Advanced AI-powered technology to determine blood groups through fingerprint patterns.
          Quick, accurate, and non-invasive testing for medical professionals.
        </p>
      </div>

      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <div 
          className={`border-2 border-dashed ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'} 
                      rounded-lg p-12 text-center transition-colors duration-200`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
        >
          <input 
            type="file" 
            ref={fileInputRef}
            className="hidden" 
            accept=".bmp,image/bmp,image/x-bmp,image/x-ms-bmp"
            onChange={handleFileChange}
          />
          
          {file ? (
            <div className="flex flex-col items-center">
              <div className="w-48 h-48 mb-4 overflow-hidden rounded-lg">
                <img 
                  src={URL.createObjectURL(file)} 
                  alt="Fingerprint preview" 
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-gray-700 font-medium">{file.name}</p>
              <button 
                onClick={handleButtonClick}
                className="mt-4 text-blue-500 hover:text-blue-700 underline"
              >
                Choose a different file
              </button>
            </div>
          ) : (
            <>
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <p className="text-gray-600 mb-4">Drag and drop your fingerprint scan here, or</p>
              <button 
                onClick={handleButtonClick}
                className="bg-blue-500 hover:bg-blue-600 text-white font-semibold px-6 py-2 rounded-lg transition-colors"
              >
                Browse Files
              </button>
              <p className="text-sm text-gray-500 mt-2">Only .bmp files are accepted</p>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      <div className="text-center">
        <button
          onClick={handleDetection}
          disabled={isLoading || !file || !!error}
          className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-8 py-3 rounded-lg text-lg transition-colors disabled:opacity-50"
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <Loader className="animate-spin mr-2" />
              Analyzing...
            </span>
          ) : (
            'Detect Blood Group'
          )}
        </button>

        {result && (
          <div className="mt-8 p-6 bg-green-50 rounded-lg">
            <h3 className="text-2xl font-bold text-green-800">Blood Group: {result.blood_group}</h3>
            <p className="text-green-600 mt-2">Confidence: {result.confidence}</p>
            <p className="text-green-600 mt-2">Analysis completed successfully</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Home;