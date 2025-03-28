import React, { useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Form, Button, Card, Alert, Spinner } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data);
    } catch (err) {
      console.error('Error details:', err.response?.data);
      setError(err.response?.data?.error || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="mt-5">
      <Row className="justify-content-center mb-4">
        <Col md={8} className="text-center">
          <h1 className="mb-3">Blood Group Detection</h1>
          <p className="lead">
            Upload a fingerprint image to predict the blood group
          </p>
        </Col>
      </Row>

      <Row className="justify-content-center">
        <Col md={6}>
          <Card className="shadow-sm">
            <Card.Body>
              <Form onSubmit={handleSubmit}>
                <Form.Group className="mb-3">
                  <Form.Label>Upload Fingerprint Image</Form.Label>
                  <Form.Control 
                    type="file" 
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                </Form.Group>
                
                <div className="d-grid gap-2">
                  <Button 
                    variant="primary" 
                    type="submit"
                    disabled={loading || !file}
                  >
                    {loading ? (
                      <>
                        <Spinner
                          as="span"
                          animation="border"
                          size="sm"
                          role="status"
                          aria-hidden="true"
                          className="me-2"
                        />
                        Processing...
                      </>
                    ) : 'Predict Blood Group'}
                  </Button>
                </div>
              </Form>

              {error && (
                <Alert variant="danger" className="mt-3">
                  {error}
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card className="shadow-sm h-100">
            <Card.Body>
              <h5 className="mb-3">Preview & Results</h5>
              
              {preview && (
                <div className="text-center mb-3">
                  <img 
                    src={preview} 
                    alt="Fingerprint Preview" 
                    className="img-fluid mb-3"
                    style={{ maxHeight: '200px' }}
                  />
                </div>
              )}
              
              {result && (
                <div className="mt-3">
                  <h5 className="text-center">Prediction Result</h5>
                  {result.error ? (
                    <Alert variant="danger">
                      {result.error}
                    </Alert>
                  ) : (
                    <div className="text-center">
                      <h2 className="text-primary mb-2">{result.blood_group}</h2>
                      <p className="text-muted">Confidence: {result.confidence}</p>
                    </div>
                  )}
                </div>
              )}
              
              {!preview && !result && (
                <div className="text-center text-muted py-5">
                  <p>Upload an image to see preview and results</p>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default App; 