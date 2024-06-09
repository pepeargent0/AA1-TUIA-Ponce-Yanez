"""import streamlit as st
import joblib
import pandas as pd

# Cargar los modelos guardados
classification_model = joblib.load('weather_clasificacion.joblib')
regression_model = joblib.load('weather_regression.joblib')

# Función para predecir si va a llover
def predict_rain(features):
    prediction = classification_model.predict(features)
    return prediction[0]

# Función para predecir la cantidad de lluvia
def predict_rainfall(features):
    prediction = regression_model.predict(features)
    return prediction[0]

# Función para mostrar los resultados
def show_results(rain_prediction, rainfall_prediction):
    st.header('Resultados')
    st.write(f'¿Va a llover? {"Sí" if rain_prediction else "No"}')
    st.write(f'Cantidad de lluvia: {rainfall_prediction:.2f} mm')

# Función principal de la aplicación
def main():
    # Título de la aplicación
    st.title('Predicción del clima')

    # Sección de entrada de características para la predicción
    st.header('Entrada de características')

    # Ejemplo de características para predecir
    example_features = {
        'MinTemp': 20.0,
        'MaxTemp': 30.0,
        'Rainfall': 5.0,
        'Evaporation': 4.0,
        'Sunshine': 8.0,
        'WindGustSpeed': 40.0,
        'PressureVariation': 5.0,
        'TempVariation': 10.0,
        'HumidityVariation': 30.0,
        'CloudVariation': 3.0,
        'WindSpeedVariation': 10.0,
        'RainToday': 1
    }

    # Recopilar las características del usuario
    user_input = {}
    for feature_name, default_value in example_features.items():
        user_input[feature_name] = st.number_input(f'{feature_name}', value=default_value)

    # Convertir las características en un DataFrame para la predicción
    features_df = pd.DataFrame([user_input])

    # Realizar las predicciones cuando se presiona el botón
    if st.button('Predecir'):
        rain_prediction = predict_rain(features_df)
        rainfall_prediction = predict_rainfall(features_df)
        show_results(rain_prediction, rainfall_prediction)

if __name__ == "__main__":
    main()
"""

import streamlit as st
import joblib
import pandas as pd




pipeline_regresion = joblib.load('weather_regression.pkl')
pipeline_clasificacion = joblib.load('weather_clasificacion.pkl')
data = pd.read_csv('weatherAUS.csv')

# Obtener las columnas esperadas por el modelo
columnas_esperadas = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'RainToday', 'PressureVariation', 'TempVariation',
       'HumidityVariation', 'CloudVariation', 'WindSpeedVariation']

# Sliders para las características
sliders = {}
for col in columnas_esperadas:
    if col == 'RainToday':

        sliders[col] = st.slider(col, int(data[col].min()), int(data[col].max()))
    else:
        sliders[col] = st.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))

# Crear un DataFrame con los valores para predecir
data_para_predecir = pd.DataFrame([sliders])

# Realizar la predicción de clasificación
prediccion_clasificacion = pipeline_clasificacion.predict(data_para_predecir)

# Realizar la predicción de regresión si clasificación indica que lloverá
prediccion_regresion = None
if prediccion_clasificacion[0] == 1:
    prediccion_regresion = pipeline_regresion.predict(data_para_predecir)

# Mostrar la predicción de clasificación
st.write('Predicción Clasificación:', 'Lloverá' if prediccion_clasificacion[0] == 1 else 'No lloverá')

# Mostrar la predicción de regresión si clasificación indica que lloverá
if prediccion_clasificacion[0] == 1:
    st.write(f'Predicción Regresión (cantidad de lluvia): {prediccion_regresion[0]} mm' if prediccion_regresion is not None else 'No disponible')
