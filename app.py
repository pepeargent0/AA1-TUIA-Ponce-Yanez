import streamlit as st
import joblib
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def create_model_regresion(num_hidden_layers=5, hidden_layer_size=285, activation='relu', optimizer='adam', dropout_rate=0.002802855543727195, learning_rate=4.048487759836856e-05):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_shape=(18,), activation=activation))
    model.add(Dropout(dropout_rate))
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(hidden_layer_size, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

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

# Título de la aplicación
st.title('Predicción del clima')

# Entrada de características para la predicción
st.header('Entrada de características')

# Ejemplo de características para predecir
example_features = {
    'MinTemp': 20.0,
    'MaxTemp': 30.0,
    'Rainfall': 5.0,
    'Evaporation': 4.0,
    'Sunshine': 8.0,
    'WindGustSpeed': 40.0,
    'WindSpeed9am': 20.0,
    'WindSpeed3pm': 30.0,
    'Humidity9am': 80.0,
    'Humidity3pm': 50.0,
    'Pressure9am': 1015.0,
    'Pressure3pm': 1010.0,
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

# Convertir las características en una matriz 2D para la predicción
features_array = np.array(list(user_input.values())).reshape(1, -1)

# Realizar las predicciones
if st.button('Predecir'):
    rain_prediction = predict_rain(features_array)
    rainfall_prediction = predict_rainfall(features_array)

    # Mostrar los resultados
    st.header('Resultados')
    st.write(f'¿Va a llover? {"Sí" if rain_prediction else "No"}')
    st.write(f'Cantidad de lluvia: {rainfall_prediction:.2f} mm')


