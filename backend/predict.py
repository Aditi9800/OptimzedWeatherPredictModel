import sys
import numpy as np
import pickle

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Parse input features from command line arguments
main_temp = float(sys.argv[1])
visibility = float(sys.argv[2])
wind_speed = float(sys.argv[3])

# Prepare the feature array for prediction
features = np.array([[main_temp, visibility, wind_speed]])

# Make prediction
prediction = model.predict(features)[0]

# Output the prediction
print(int(prediction))
