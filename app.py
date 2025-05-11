crop_dict = {
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


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         features = [
#             float(request.form['nitrogen']),
#             float(request.form['phosphorus']),
#             float(request.form['potassium']),
#             float(request.form['temperature']),
#             float(request.form['humidity']),
#             float(request.form['ph']),
#             float(request.form['rainfall'])
#         ]

#         #When scaling is required use below snippet
#         input_data = np.array([features])
#         input_scaled = scaler.transform(input_data)  # Apply scaling

#         predicted_class = model.predict(input_scaled)[0]  # Use scaled input
#         prediction = crop_dict.get(predicted_class, f"Unknown (label {predicted_class})")

#         #When there is no scaling for tree based algos use the below snippet:
#         # input_data = np.array([features])
#         # # prediction = model.predict(input_data)[0]
#         # predicted_class = model.predict(input_data)[0]
#         # prediction = crop_dict.get(predicted_class, f"Unknown (label {predicted_class})")


#         # if hasattr(model, "predict_proba"):
#         #     confidence = np.max(model.predict_proba(input_data)) * 100
#         # else:
#         #     confidence = None

#         return render_template(
#             'index.html', 
#             prediction=prediction, 
#             # confidence=confidence,
#             form_data=request.form
#         )
#     except Exception as e:
#         return render_template('index.html', error=str(e))

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

        predicted_class = model.predict(input_scaled)[0]  # This returns a number
        crop_name = crop_dict.get(predicted_class, f"Unknown (label {predicted_class})")
        
        # Define image filename based on the crop name
        crop_image = f"{crop_name.lower()}.jpg"

        return render_template(
            'index.html', 
            prediction=crop_name,
            crop_image=crop_image,
            form_data=request.form
        )
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
