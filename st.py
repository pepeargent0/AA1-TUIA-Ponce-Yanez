import streamlit as st
import joblib
import numpy as np

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
    return result

# Interfaz de usuario de Streamlit
st.title('Predicción de lluvia en Australia')

if modelo_cuanto_llueve and modelo_si_llueve:
    with st.form(key='prediction_form'):
        MinTemp = st.number_input('MinTemp', value=0.0)
        MaxTemp = st.number_input('MaxTemp', value=0.0)
        Rainfall = st.number_input('Rainfall', value=0.0)
        Evaporation = st.number_input('Evaporation', value=0.0)
        Sunshine = st.number_input('Sunshine', value=0.0)
        WindGustSpeed = st.number_input('WindGustSpeed', value=0.0)
        Pressure9am = st.number_input('Pressure9am', value=0.0)
        Pressure3pm = st.number_input('Pressure3pm', value=0.0)
        Temp9am = st.number_input('Temp9am', value=0.0)
        Temp3pm = st.number_input('Temp3pm', value=0.0)
        Humidity9am = st.number_input('Humidity9am', value=0.0)
        Humidity3pm = st.number_input('Humidity3pm', value=0.0)
        Cloud9am = st.number_input('Cloud9am', value=0.0)
        Cloud3pm = st.number_input('Cloud3pm', value=0.0)
        WindSpeed9am = st.number_input('WindSpeed9am', value=0.0)
        WindSpeed3pm = st.number_input('WindSpeed3pm', value=0.0)
        RainToday = st.selectbox('RainToday', options=[0, 1])

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
            'RainToday': RainToday
        }
        result = predict(data)
        st.success(f'¿Es probable que llueva? {result}')
else:
    st.error("Modelos no cargados correctamente. Por favor, contacte al administrador.")

