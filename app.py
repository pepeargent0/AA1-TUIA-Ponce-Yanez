from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import numpy as np

# Cargar los modelos
modelo_cuanto_llueve = joblib.load('weather_regression.pkl')
modelo_si_llueve = joblib.load('weather_clasificacion.pkl')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Función ficticia para la autenticación
def authenticate(username, password):
    # Aquí puedes agregar la lógica de autenticación
    return username == "admin" and password == "admin"


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if authenticate(username, password):
            return redirect(url_for('predict'))
        else:
            flash('Credenciales incorrectas. Por favor, inténtalo de nuevo.')
    return render_template('login.html')

@app.route('/prediccion', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        example_features = {
            'MinTemp': float(data['MinTemp']),
            'MaxTemp': float(data['MaxTemp']),
            'Rainfall': float(data['Rainfall']),
            'Evaporation': float(data['Evaporation']),
            'Sunshine': float(data['Sunshine']),
            'WindGustSpeed': float(data['WindGustSpeed']),
            'Pressure9am': float(data['Pressure9am']),
            'Pressure3pm': float(data['Pressure3pm']),
            'Temp9am': float(data['Temp9am']),
            'Temp3pm': float(data['Temp3pm']),
            'Humidity9am': float(data['Humidity9am']),
            'Humidity3pm': float(data['Humidity3pm']),
            'Cloud9am': float(data['Cloud9am']),
            'Cloud3pm': float(data['Cloud3pm']),
            'WindSpeed9am': float(data['WindSpeed9am']),
            'WindSpeed3pm': float(data['WindSpeed3pm']),
            'RainToday': float(data['RainToday'])
        }

        data_transform = {
            'MinTemp': example_features['MinTemp'],
            'MaxTemp': example_features['MaxTemp'],
            'Rainfall': example_features['Rainfall'],
            'Evaporation': example_features['Evaporation'],
            'Sunshine': example_features['Sunshine'],
            'WindGustSpeed': example_features['WindGustSpeed'],
            'PressureVariation': example_features['Pressure3pm'] - example_features['Pressure9am'],
            'TempVariation': example_features['Temp3pm'] - example_features['Temp9am'],
            'HumidityVariation': example_features['Humidity3pm'] - example_features['Humidity9am'],
            'CloudVariation': example_features['Cloud3pm'] - example_features['Cloud9am'],
            'WindSpeedVariation': example_features['WindSpeed3pm'] - example_features['WindSpeed9am'],
            'RainToday': example_features['RainToday']
        }

        features_array = np.array(list(data_transform.values())).reshape(1, -1)
        prediccion_llueve = modelo_si_llueve.predict(features_array)
        if prediccion_llueve == 0:
            result = 'No'
        else:
            predicted_rainfall = modelo_cuanto_llueve.predict(features_array)
            result = f'Si, {predicted_rainfall[0][0]:.2f} mm'
        return jsonify(result=result)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
