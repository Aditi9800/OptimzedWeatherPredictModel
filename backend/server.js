const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

let model;

// Load the trained model
try {
  const modelPath = path.join(__dirname, 'model.pkl');

  model = fs.readFileSync(modelPath);
  console.log('Model loaded successfully');
} catch (error) {
  console.error('Error loading model:', error);
  model = null;
}

app.post('/predict', (req, res) => {  
  const { main_temp, visibility, wind_speed } = req.body;

  // Validate input data
  if (main_temp == null || visibility == null || wind_speed == null) {
    return res.status(400).json({ error: 'Invalid input' });
  }

  // Format input data for prediction
  const features = [main_temp, visibility, wind_speed];

  // Use Python to load the model and make the prediction
  const options = {
    mode: 'text',
    pythonOptions: ['-u'],
    scriptPath: __dirname,
    args: features,
  };

  PythonShell.run('predict.py', options).then(results => {
    const prediction = results[0];
    res.json({ prediction: parseInt(prediction, 10) });
  }).catch(err => {
    console.error('Error:', err);
    res.status(500).json({ error: 'Error making prediction' });
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
