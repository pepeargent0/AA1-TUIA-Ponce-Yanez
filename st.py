import streamlit as st
import joblib
import numpy as np

# Cargar modelos
@st.cache_data
def load_models():
    try:
        modelo_cuanto_llueve = joblib.load('weather_regression.pkl')
    except FileNotFoundError:
        st.error("Error: 'weather_regression.pkl' no encontrado.")
        modelo_cuanto_llueve = None
    except Exception as e:
        st.error(f"Error cargando 'weather_regression.pkl': {e}")
        modelo_cuanto_llueve = None

    try:
        modelo_si_llueve = joblib.load('weather_clasificacion.pkl')
    except FileNotFoundError:
        st.error("Error: 'weather_clasificacion.pkl' no encontrado.")
        modelo_si_llueve = None
    except Exception as e:
        st.error(f"Error cargando 'weather_clasificacion.pkl': {e}")
        modelo_si_llueve = None

    return modelo_cuanto_llueve, modelo_si_llueve

modelo_cuanto_llueve, modelo_si_llueve = load_models()

# Función de predicción
def predict(data):
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
        'RainToday': float(data['RainToday']),
        'WindGustDir': data['WindGustDir'],
        'WindDir9am': data['WindDir9am'],
        'WindDir3pm': data['WindDir3pm']
    }

    wind_dir_mapping = {
        'N': 0, 'NNE': 1, 'NE': 2, 'ENE': 3, 'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7,
        'S': 8, 'SSW': 9, 'SW': 10, 'WSW': 11, 'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15
    }
    example_features['WindGustDir'] = wind_dir_mapping[example_features['WindGustDir']]
    example_features['WindDir9am'] = wind_dir_mapping[example_features['WindDir9am']]
    example_features['WindDir3pm'] = wind_dir_mapping[example_features['WindDir3pm']]
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
    return result

# Interfaz de usuario de Streamlit
st.title('Predicción de lluvia en Australia')

if modelo_cuanto_llueve and modelo_si_llueve:
    with st.form(key='prediction_form'):
        Location = st.selectbox('Location', ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'Melbourne Airport', 'Mount Gambier', 'Sydney', 'Sydney Airport'])
        MinTemp = st.slider('MinTemp', min_value=-8.5, max_value=33.9, value=19.7, step=0.1)
        MaxTemp = st.slider('MaxTemp', min_value=-4.8, max_value=48.1, value=21.5, step=0.1)
        Rainfall = st.slider('Rainfall', min_value=0.0, max_value=371.0, value=0.0, step=0.1)
        Evaporation = st.slider('Evaporation', min_value=0.0, max_value=145.0, value=13.2, step=0.1)
        Sunshine = st.slider('Sunshine', min_value=0.0, max_value=14.5, value=1.3, step=0.1)
        WindGustSpeed = st.slider('WindGustSpeed', min_value=6.0, max_value=135.0, value=67.0, step=1.0)
        Pressure9am = st.slider('Pressure9am', min_value=980.5, max_value=1041.0, value=1029.6, step=0.1)
        Pressure3pm = st.slider('Pressure3pm', min_value=977.1, max_value=1039.6, value=1033.6, step=0.1)
        Temp9am = st.slider('Temp9am', min_value=-7.2, max_value=40.2, value=20.3, step=0.1)
        Temp3pm = st.slider('Temp3pm', min_value=-5.4, max_value=46.7, value=19.0, step=0.1)
        Humidity9am = st.slider('Humidity9am', min_value=0.0, max_value=100.0, value=61.0, step=0.1)
        Humidity3pm = st.slider('Humidity3pm', min_value=0.0, max_value=100.0, value=56.0, step=0.1)
        Cloud9am = st.slider('Cloud9am', min_value=0.0, max_value=9.0, value=5.0, step=0.1)
        Cloud3pm = st.slider('Cloud3pm', min_value=0.0, max_value=9.0, value=7.0, step=0.1)
        WindSpeed9am = st.slider('WindSpeed9am', min_value=0.0, max_value=130.0, value=39.0, step=0.1)
        WindSpeed3pm = st.slider('WindSpeed3pm', min_value=0.0, max_value=87.0, value=26.0, step=0.1)
        RainToday = st.selectbox('RainToday', ['No', 'Sí'])
        RainToday = 1 if RainToday == 'Sí' else 0
        WindGustDir = st.selectbox('WindGustDir', ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        WindDir9am = st.selectbox('WindDir9am', ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        WindDir3pm = st.selectbox('WindDir3pm', ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])

        submit_button = st.form_submit_button(label='Predecir')

    if submit_button:
        data = {
            'MinTemp': MinTemp,
            'MaxTemp': MaxTemp,
            'Rainfall': Rainfall,
            'Evaporation': Evaporation,
            'Sunshine': Sunshine,
            'WindGustSpeed': WindGustSpeed,
            'Pressure9am': Pressure9am,
            'Pressure3pm': Pressure3pm,
            'Temp9am': Temp9am,
            'Temp3pm': Temp3pm,
            'Humidity9am': Humidity9am,
            'Humidity3pm': Humidity3pm,
            'Cloud9am': Cloud9am,
            'Cloud3pm': Cloud3pm,
            'WindSpeed9am': WindSpeed9am,
            'WindSpeed3pm': WindSpeed3pm,
            'RainToday': RainToday,
            'WindGustDir': WindGustDir,
            'WindDir9am': WindDir9am,
            'WindDir3pm': WindDir3pm
        }
        result = predict(data)
        st.success(f'¿Es probable que llueva? {result}')
else:
    st.error("Modelos no cargados correctamente. Por favor, contacte al administrador.")
