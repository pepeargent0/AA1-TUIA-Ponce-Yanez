import joblib
import numpy as np

modelo_cuanto_llueve = joblib.load('weather_regression.joblib')
modelo_si_llueve = joblib.load('weather_clasificacion.joblib')

example_features = {
    'Location': 'DDDD',
    'MinTemp': 0.1,
    'MaxTemp': 20,
    'Rainfall': 30,
    'Evaporation': 30,
    'Sunshine': 50,
    'WindGustDir': 60,
    'WindGustSpeed': 40,
    'WindDir9am': 50,
    'WindDir3pm': 50,
    'WindSpeed9am': 30,
    'WindSpeed3pm': 20,
    'Humidity9am': 10,
    'Humidity3pm': 20,
    'Pressure9am': 30,
    'Pressure3pm': 20,
    'Cloud9am': 10,
    'Cloud3pm': 10,
    'Temp9am': 20,
    'Temp3pm': 30,
    'RainToday': 10
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
    print(f' LLueve: No')
else:
    print(f' LLueve: SI')
    predicted_rainfall = modelo_cuanto_llueve.predict(features_array)
    print(f'Cantidad de lluvia predicha: {predicted_rainfall[0][0]:.2f} mm')
