import joblib
import numpy as np

# Cargar el pipeline de regresión guardado
loaded_pipeline = joblib.load('weather_clasificacion.joblib')



# Crear un ejemplo de entrada para la predicción
example_features = {
    'MinTemp': 1.0,
    'MaxTemp': 10.0,
    'Rainfall': 1.0,
    'Evaporation': 40.0,
    'Sunshine': 0.0,
    'WindGustSpeed': 40.0,
    'PressureVariation': 50.0,
    'TempVariation': -10.0,
    'HumidityVariation': 90.0,
    'CloudVariation': 3.0,
    'WindSpeedVariation': 10.0,
    'RainToday': 0
}

# Convertir las características en una matriz 2D para la predicción
features_array = np.array(list(example_features.values())).reshape(1, -1)

# Realizar la predicción
predicted_rainfall = loaded_pipeline.predict(features_array)

# Mostrar el resultado
print(f'Cantidad de lluvia predicha: {predicted_rainfall[0]:.2f} mm')
