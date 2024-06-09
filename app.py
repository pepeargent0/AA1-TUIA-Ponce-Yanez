import streamlit as st
import joblib
import pandas as pd


# Función para verificar las credenciales
def authenticate(username, password):
    # Aquí deberías realizar la autenticación, por ejemplo, comparando con credenciales almacenadas
    # En este ejemplo, solo se verifica que el usuario y la contraseña no estén vacíos
    if username != "" and password != "":
        return True
    else:
        return False


# Función para cargar los modelos guardados
def load_models():
    classification_model = joblib.load('weather_clasificacion.joblib')
    regression_model = joblib.load('weather_regression.joblib')
    return classification_model, regression_model


# Función para predecir si va a llover
def predict_rain(features, classification_model):
    prediction = classification_model.predict(features)
    return prediction[0]


# Función para predecir la cantidad de lluvia
def predict_rainfall(features, regression_model):
    prediction = regression_model.predict(features)
    return prediction[0]


# Función para mostrar los resultados
def show_results(rain_prediction, rainfall_prediction):
    st.header('Resultados')
    st.write(f'¿Va a llover? {"Sí" if rain_prediction else "No"}')
    st.write(f'Cantidad de lluvia: {rainfall_prediction:.2f} mm')


# Función principal de la aplicación
def main():
    # Cargar el archivo CSS
    st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)

    # Título de la aplicación
    st.title('Predicción del clima')

    # Div contenedor para el formulario de inicio de sesión
    st.markdown('<div class="container login-container">', unsafe_allow_html=True)

    # Card para el formulario de inicio de sesión
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)

    # Sección de inicio de sesión
    st.header('Inicio de sesión')
    username = st.text_input('Usuario')
    password = st.text_input('Contraseña', type='password')

    # Verificar si se ha iniciado sesión
    if st.button('Iniciar sesión'):
        if authenticate(username, password):
            st.success('Inicio de sesión exitoso!')
            # Cargar los modelos después de la autenticación
            classification_model, regression_model = load_models()
            # Pasar a la pantalla de predicción
            predict_weather(classification_model, regression_model)
        else:
            st.error('Credenciales incorrectas. Por favor, inténtalo de nuevo.')

    # Cerrar el card y el contenedor
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Función para la pantalla de predicción del clima
def predict_weather(classification_model, regression_model):
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
        rain_prediction = predict_rain(features_df, classification_model)
        rainfall_prediction = predict_rainfall(features_df, regression_model)
        show_results(rain_prediction, rainfall_prediction)


if __name__ == "__main__":
    main()
