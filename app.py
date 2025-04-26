from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and label encoder
model = None
label_encoder = None
try:
    model = joblib.load('equipment_failure/random_forest_model.pkl')
    label_encoder = joblib.load('equipment_failure/label_encoder.pkl')
except:
    print("Model files not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        type_val = int(request.form['type'])
        air_temp = float(request.form['air_temperature'])
        process_temp = float(request.form['process_temperature'])
        rotational_speed = float(request.form['rotational_speed'])
        torque = float(request.form['torque'])
        tool_wear = float(request.form['tool_wear'])
        
        # Calculate implicit features (example calculations - adjust as needed)
        # Feature 1: Temperature difference (process - air)
        temp_difference = process_temp - air_temp
        
        # Feature 2: Power (rotational speed * torque)
        power = rotational_speed * torque
        
        # Create features list with all 8 features in correct order
        features = [
            type_val,
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear,
            temp_difference,  # First implicit feature
            power             # Second implicit feature
        ]
        
        # Make prediction
        if model is not None and label_encoder is not None:
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][1]  # Probability of failure
            
            # Convert numeric prediction back to label
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            
            return jsonify({
                'prediction': prediction_label,
                'probability': float(probability),
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True) 