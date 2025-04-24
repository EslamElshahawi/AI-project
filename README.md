# Equipment Failure Prediction System

This web application predicts equipment failures based on various parameters such as temperature, pressure, vibration, operating hours, and maintenance frequency.

## Features

- Real-time equipment failure prediction
- Modern, responsive user interface
- Probability-based predictions
- Easy-to-use input form

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
- if this doesn't work install every dependecy seperetly from the requirements.txt using

```bash
pip install flask numpy pandas scikit-learn joblib gunicorn python-dotenv
```

3. Train the model:
- Use the provided Jupyter notebook (`equipmentfailure.ipynb`) to train your model
- Save the trained model as `model/equipment_failure_model.pkl`

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Input Parameters

- Temperature (°C): Current operating temperature
- Pressure (PSI): System pressure
- Vibration (mm/s): Equipment vibration level
- Operating Hours: Total hours of operation
- Maintenance Frequency (days): Days between maintenance

## Model Training

The model should be trained using historical equipment data with the following features:
Quality Type variants
Air Temperature (K)
Process Temperature (K)
Rotational Speed (rpm)
Torque (Nm)
Tool Wear (minutes)

## License

MIT License 