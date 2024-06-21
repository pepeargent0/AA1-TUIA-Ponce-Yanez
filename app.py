from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import joblib
import numpy as np
from functools import wraps

app = Flask(__name__)
app.secret_key = 'AA1_2024.'

try:
    modelo_cuanto_llueve = joblib.load('weather_regression.pkl')
except FileNotFoundError:
    modelo_cuanto_llueve = None
    app.logger.error("Error: 'weather_regression.pkl' no encontrado.")
except Exception as e:
    modelo_cuanto_llueve = None
    app.logger.error(f"Error cargando 'weather_regression.pkl': {e}")

try:
    modelo_si_llueve = joblib.load('weather_clasificacion.pkl')
except FileNotFoundError:
    modelo_si_llueve = None
    app.logger.error("Error: 'weather_clasificacion.pkl' no encontrado.")
except Exception as e:
    modelo_si_llueve = None
    app.logger.error(f"Error cargando 'weather_clasificacion.pkl': {e}")


def authenticate(username, password):
    if username == "admin" and password == "admin":
        session['logged_in'] = True
        return True
    return False


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


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
@login_required
def predict():
    if not modelo_cuanto_llueve or not modelo_si_llueve:
        return jsonify(result="Modelos no cargados correctamente. Por favor, contacte al administrador."), 500

    if request.method == 'POST':
        try:
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
        except Exception as e:
            app.logger.error(f"Error processing prediction: {e}")
            return jsonify(result="Error procesando la predicción"), 500
    return render_template('predict.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


@app.errorhandler(404)
def page_not_found(erro_404):
    app.logger.error(f"Error: Pagina no encontrada {erro_404}")
    return redirect(url_for('login'))


@app.errorhandler(500)
def internal_server_error(erro_500):
    app.logger.error(f"Error: Pagina no encontrada {erro_500}")
    return redirect(url_for('login'))


@app.errorhandler(Exception)
def handle_exception(exception):
    app.logger.error(f"Unhandled exception: {exception}")
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
