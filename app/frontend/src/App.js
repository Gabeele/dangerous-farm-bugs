import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setImagePreview(null); // Clear previous image preview

    // Preview the selected image
    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(process.env.REACT_APP_API_URL + '/predict', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data.prediction);
      } else {
        throw new Error('Failed to fetch');
      }
    } catch (error) {
      setResult('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🐛 Dangerous Farm Bug Detector 🦟</h1>
        <p>Keep your farm crop safe! Upload an image to detect if it contains a dangerous bug.</p>
        <form onSubmit={handleSubmit}>
          <input type="file" accept=".jpg, .jpeg" onChange={handleFileChange} />
          <button type="submit" disabled={loading}>{loading ? 'Detecting...' : 'Detect'}</button>
        </form>
        {imagePreview && (
          <div>
            <h6>Uploaded Image Preview</h6>
            <img src={imagePreview} alt="Uploaded Preview" style={{ maxWidth: '100%', maxHeight: '200px' }} />
          </div>
        )}
        {result && <p>Result: {result}</p>}
      </header>
    </div>
  );
}

export default App;
