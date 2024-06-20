import numpy as np
import streamlit as st
import joblib
import pandas as pd


def authenticate(username, password):
    return username == "admin" and password == "admin"


# Función para cargar los modelos guardados
def load_models():
    try:
        classification_model = joblib.load('weather_clasificacion.joblib')
        regression_model = joblib.load('weather_regression.joblib')
        return classification_model, regression_model
    except FileNotFoundError:
        st.error(
            "No se encontraron los archivos de los modelos. Asegúrate de que 'weather_clasificacion.joblib' y 'weather_regression.joblib' existan.")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None


# Función para predecir si va a llover
def predict_rain(features, classification_model):
    try:
        prediction = classification_model.predict(features)
        return prediction[0]
    except Exception as e:
        st.error(f"Error al realizar la predicción de clasificación: {e}")
        return None


# Función para predecir la cantidad de lluvia
def predict_rainfall(features, regression_model):
    try:
        prediction = regression_model.predict(features)
        return prediction[0]
    except Exception as e:
        st.error(f"Error al realizar la predicción de regresión: {e}")
        return None


# Función para mostrar los resultados
def show_results(rain_prediction, rainfall_prediction):
    st.header('Resultados')
    if rain_prediction is not None:
        st.write(f'¿Va a llover? {"Sí" if rain_prediction else "No"}')
    else:
        st.write("No se pudo determinar si va a llover.")

    if rainfall_prediction is not None:
        st.write(f'Cantidad de lluvia: {rainfall_prediction:.2f} mm')
    else:
        st.write("No se pudo determinar la cantidad de lluvia.")


# Función principal de la aplicación
def main():
    # Cargar el archivo CSS
    try:
        st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Archivo CSS 'styles.css' no encontrado.")

    # Título de la aplicación
    st.title('Predicción del clima')

    # Obtener la ruta de la página actual
    url = st.experimental_get_query_params()
    url_path = url['page'][0] if 'page' in url else 'login'

    # Enrutamiento básico
    if url_path == 'login':
        login_page()
    elif url_path == 'clima':
        clima_page()
    else:
        st.error('Página no encontrada.')


# Función para la página de inicio de sesión
def login_page():
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
            st.experimental_set_query_params(page='clima')
        else:
            st.error('Credenciales incorrectas. Por favor, inténtalo de nuevo.')

    # Cerrar el card y el contenedor
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Función para la página de predicción del clima
def clima_page():
    # Cargar los modelos
    classification_model, regression_model = load_models()

    if classification_model is None or regression_model is None:
        st.error("Error al cargar los modelos. No se pueden realizar predicciones.")
        return

    # Sección de entrada de características para la predicción
    st.header('Entrada de características')

    # Ejemplo de características para predecir
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

    # Recopilar las características del usuario
    user_input = {}
    for feature_name, default_value in example_features.items():
        user_input[feature_name] = st.number_input(f'{feature_name}', value=default_value)

    # Convertir las características en un DataFrame para la predicción
    features_df = pd.DataFrame([user_input])

    # Asegurarse de que las columnas están en el mismo orden que durante el entrenamiento
    features_df = features_df[list(example_features.keys())]

    # Imprimir las columnas del DataFrame y las esperadas por el modelo
    st.write("Características de entrada:", features_df.columns.tolist())

    # Obtener los nombres de las características esperadas por el modelo
    expected_columns = classification_model.feature_names_in_

    # Imprimir los nombres de las características esperadas por el modelo
    st.write("Características esperadas por el modelo:", expected_columns.tolist())

    # Verificar las diferencias entre las características
    missing_features = set(expected_columns) - set(features_df.columns)
    extra_features = set(features_df.columns) - set(expected_columns)

    if missing_features:
        st.write("Características faltantes:", list(missing_features))
    if extra_features:
        st.write("Características adicionales no esperadas:", list(extra_features))

    # Realizar las predicciones cuando se presiona el botón
    if st.button('Predecir'):
        rain_prediction = predict_rain(features_df, classification_model)
        rainfall_prediction = predict_rainfall(features_df, regression_model)
        show_results(rain_prediction, rainfall_prediction)


if __name__ == "__main__":
    main()
