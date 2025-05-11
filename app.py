import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model from the models folder
model_path = os.path.join('models', 'ExtraTreesClassifier.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the scaler
scaler_path = os.path.join('models', 'scaler.pkl')
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load crop information from CSV
crop_info_path = 'Updated_Crop_Info.csv'
crop_info_df = pd.read_csv(crop_info_path)
# Convert DataFrame to dictionary for easy access
crop_info = {}
for _, row in crop_info_df.iterrows():
    crop_name = row['Crop'].lower()
    crop_info[crop_name] = {
        'diseases': row['Diseases'],
        'irrigation': row['Irrigation'],
        'fertilizers': row['Fertilizers'],
        'pests': row['Pests'],
        'pesticides': row['Pesticides']
    }

label_map = {
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)  # Apply scaling

        predicted_class = model.predict(input_scaled)[0]  # Use scaled input
        crop_name = label_map.get(predicted_class, f"Unknown (label {predicted_class})")
        
        # Get crop information if available
        crop_details = crop_info.get(crop_name.lower(), None)
        
        return render_template(
            'index.html', 
            prediction=crop_name,
            crop_image=f"{crop_name.lower()}.jpg",
            crop_details=crop_details,
            form_data=request.form
        )
    except Exception as e:
        return render_template('index.html', error=str(e))

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
