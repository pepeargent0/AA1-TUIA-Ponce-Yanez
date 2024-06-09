"""
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, f1_score
from scikeras.wrappers import KerasRegressor

import numpy as np

import joblib


class ModeloLluvia:
    def __init__(self, file_path):
        self.df = self._leer_dataset(file_path)
        self.train = None
        self.test = None
        self.y_train_regresion = None
        self.y_test_regresion = None
        self.y_train_clasification = None
        self.y_test_clasification = None
        self.X_train = None
        self.X_test = None

    @staticmethod
    def _leer_dataset(file_path):
        df = pd.read_csv(file_path, sep=',')
        return df

    def _filtro_previo(self):
        ciudades = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier',
                    'Sydney', 'SydneyAirport']
        self.df = self.df[self.df['Location'].isin(ciudades)]
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by='Date')

    def _seprar_test_train(self):
        date_train_limit = pd.to_datetime('2015-10-06')
        self.train = self.df[self.df['Date'] <= date_train_limit]
        self.test = self.df[self.df['Date'] > date_train_limit]

    def _limpieza_train(self):
        self.train.drop(columns=['Unnamed: 0', 'Location'], inplace=True, axis=1)
        print(self.train.columns)
        self.train.sort_values(by='Date', inplace=True)
        self.train.fillna(method='ffill', inplace=True)

        []
        self.train['PressureVariation'] = self.train['Pressure3pm'] - self.train['Pressure9am']
        self.train['TempVariation'] = self.train['Temp3pm'] - self.train['Temp9am']
        self.train['HumidityVariation'] = self.train['Humidity3pm'] - self.train['Humidity9am']
        self.train['CloudVariation'] = self.train['Cloud3pm'] - self.train['Cloud9am']
        self.train['WindSpeedVariation'] = self.train['WindSpeed3pm'] - self.train['WindSpeed9am']
        self.train['RainToday'] = self.train['RainToday'].map({'No': 0, 'Yes': 1})
        self.train['RainTomorrow'] = self.train['RainTomorrow'].map({'No': 0, 'Yes': 1})

    def _limpieza_test(self):
        # Mapear valores de 'RainToday' y 'RainTomorrow'
        self.test['RainToday'] = self.test['RainToday'].map({'No': 0, 'Yes': 1})
        self.test['RainTomorrow'] = self.test['RainTomorrow'].map({'No': 0, 'Yes': 1})

        # Ordenar por 'Date'
        self.test.sort_values(by='Date', inplace=True)

        # Rellenar NaNs hacia adelante y hacia atrás
        for column in self.test.columns:
            self.test[column] = self.test[column].ffill()
            self.test[column] = self.test[column].bfill()

        # Eliminar columnas no necesarias
        self.test = self.test.drop(columns=['Unnamed: 0', 'Location', 'Date'], errors='ignore', axis=1)

        self.test['PressureVariation'] = self.test['Pressure3pm'] - self.test['Pressure9am']
        self.test['TempVariation'] = self.test['Temp3pm'] - self.test['Temp9am']
        self.test['HumidityVariation'] = self.test['Humidity3pm'] - self.test['Humidity9am']
        self.test['CloudVariation'] = self.test['Cloud3pm'] - self.test['Cloud9am']
        self.test['WindSpeedVariation'] = self.test['WindSpeed3pm'] - self.test['WindSpeed9am']

    def crear(self):
        self._filtro_previo()
        self._seprar_test_train()
        self._limpieza_train()
        self._limpieza_test()
        self.train = self.train.drop(columns=['Date'])
        columns_to_aggregate = ['Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Humidity9am',
                                'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'WindSpeed3pm', 'WindSpeed9am', 'WindGustDir',
                                'WindDir9am', 'WindDir3pm']
        self.train.drop(columns=columns_to_aggregate, inplace=True, axis=1)
        self.test.drop(columns=columns_to_aggregate, inplace=True, axis=1)

        self.X_train = self.train.drop(columns=['RainTomorrow', 'RainfallTomorrow'])
        self.y_train_regresion = self.train['RainfallTomorrow']
        self.X_test = self.test.drop(columns=['RainTomorrow', 'RainfallTomorrow'])
        self.y_test_regresion = self.test['RainfallTomorrow']
        self.y_train_clasification = self.train['RainTomorrow']
        self.y_test_clasification = self.test['RainTomorrow']
        return (self.y_train_regresion, self.y_test_regresion, self.y_train_clasification, self.y_test_clasification,
                self.X_train, self.X_test)


def show_metrics_regresion(y, y_pred, title, nr_neuronal=True):
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(title)
    print("Mean Squared Error :", mse)
    print("R-squared:", r2)
    print("Mean Absolute Error (MAE):", mae)
    if nr_neuronal:
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        print("Mean Absolute Percentage Error (MAPE):", mape)


modelo_lluvia = ModeloLluvia('weatherAUS.csv')
(y_train_regresion, y_test_regresion, y_train_clasification, y_test_clasification,
 X_train, X_test) = modelo_lluvia.crear()

print(X_train.columns)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

model = Sequential()
model.add(Dense(32, input_shape=(12,), activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train_regresion, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

show_metrics_regresion(y_train_regresion, y_pred_train, "Métricas del conjunto de entrenamiento:", False)
show_metrics_regresion(y_test_regresion, y_pred_test, "Métricas del conjunto de Prueba:", False)
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModeloLluvia:
    def __init__(self, file_path):
        self.df = self._leer_dataset(file_path)
        self.train = None
        self.test = None
        self.y_train_regresion = None
        self.y_test_regresion = None
        self.y_train_clasificacion = None
        self.y_test_clasificacion = None
        self.X_train = None
        self.X_test = None

    @staticmethod
    def _leer_dataset(file_path):
        df = pd.read_csv(file_path, sep=',')
        return df

    def _filtro_previo(self):
        ciudades = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier',
                    'Sydney', 'SydneyAirport']
        self.df = self.df[self.df['Location'].isin(ciudades)]
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by='Date')

    def _separar_test_train(self):
        date_train_limit = pd.to_datetime('2015-10-06')
        self.train = self.df[self.df['Date'] <= date_train_limit]
        self.test = self.df[self.df['Date'] > date_train_limit]

    def _limpieza_train(self):
        self.train.drop(columns=['Unnamed: 0', 'Location'], inplace=True, axis=1)
        self.train.sort_values(by='Date', inplace=True)
        self.train.fillna(method='ffill', inplace=True)

        self.train['PressureVariation'] = self.train['Pressure3pm'] - self.train['Pressure9am']
        self.train['TempVariation'] = self.train['Temp3pm'] - self.train['Temp9am']
        self.train['HumidityVariation'] = self.train['Humidity3pm'] - self.train['Humidity9am']
        self.train['CloudVariation'] = self.train['Cloud3pm'] - self.train['Cloud9am']
        self.train['WindSpeedVariation'] = self.train['WindSpeed3pm'] - self.train['WindSpeed9am']
        self.train['RainToday'] = self.train['RainToday'].map({'No': 0, 'Yes': 1})
        self.train['RainTomorrow'] = self.train['RainTomorrow'].map({'No': 0, 'Yes': 1})

    def _limpieza_test(self):
        self.test['RainToday'] = self.test['RainToday'].map({'No': 0, 'Yes': 1})
        self.test['RainTomorrow'] = self.test['RainTomorrow'].map({'No': 0, 'Yes': 1})
        self.test.sort_values(by='Date', inplace=True)

        for column in self.test.columns:
            self.test[column] = self.test[column].ffill()
            self.test[column] = self.test[column].bfill()

        self.test = self.test.drop(columns=['Unnamed: 0', 'Location', 'Date'], errors='ignore', axis=1)
        self.test['PressureVariation'] = self.test['Pressure3pm'] - self.test['Pressure9am']
        self.test['TempVariation'] = self.test['Temp3pm'] - self.test['Temp9am']
        self.test['HumidityVariation'] = self.test['Humidity3pm'] - self.test['Humidity9am']
        self.test['CloudVariation'] = self.test['Cloud3pm'] - self.test['Cloud9am']
        self.test['WindSpeedVariation'] = self.test['WindSpeed3pm'] - self.test['WindSpeed9am']

    def crear(self):
        self._filtro_previo()
        self._separar_test_train()
        self._limpieza_train()
        self._limpieza_test()
        self.train = self.train.drop(columns=['Date'])
        columns_to_aggregate = ['Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Humidity9am',
                                'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'WindSpeed3pm', 'WindSpeed9am', 'WindGustDir',
                                'WindDir9am', 'WindDir3pm']
        self.train.drop(columns=columns_to_aggregate, inplace=True, axis=1)
        self.test.drop(columns=columns_to_aggregate, inplace=True, axis=1)

        self.X_train = self.train.drop(columns=['RainTomorrow', 'RainfallTomorrow'])
        self.y_train_regresion = self.train['RainfallTomorrow']
        self.X_test = self.test.drop(columns=['RainTomorrow', 'RainfallTomorrow'])
        self.y_test_regresion = self.test['RainfallTomorrow']
        self.y_train_clasificacion = self.train['RainTomorrow']
        self.y_test_clasificacion = self.test['RainTomorrow']

        return (self.y_train_regresion, self.y_test_regresion, self.y_train_clasificacion, self.y_test_clasificacion,
                self.X_train, self.X_test)

def create_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(12,), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Crear el modelo de keras envuelto en KerasRegressor
keras_regressor = KerasRegressor(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

# Crear el objeto ModeloLluvia y generar los conjuntos de datos
modelo_lluvia = ModeloLluvia('weatherAUS.csv')
(y_train_regresion, y_test_regresion, y_train_clasificacion, y_test_clasificacion, X_train, X_test) = modelo_lluvia.crear()

# Definir el preprocesamiento solo para las características numéricas, ya que las categóricas ya están preprocesadas en la clase
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Preprocesamiento para datos numéricos
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combinar preprocesamientos
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Crear el pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', keras_regressor)
])

# Entrenar el pipeline
pipeline.fit(X_train, y_train_regresion)

# Hacer predicciones
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Función para mostrar métricas de regresión
def show_metrics_regresion(y_true, y_pred, mensaje, verbose=True):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if verbose:
        print(mensaje)
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R2: {r2:.4f}')
    return mse, rmse, mae, r2

# Mostrar métricas
show_metrics_regresion(y_train_regresion, y_pred_train, "Métricas del conjunto de entrenamiento:", False)
show_metrics_regresion(y_test_regresion, y_pred_test, "Métricas del conjunto de Prueba:", False)

"""




from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier



best_params = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.005958432842192192,
    'learning_rate_init': 0.0001422171191037891
}

pipeline_clasificacion = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', MLPClassifier(**best_params, max_iter=2000))
])
pipeline_clasificacion.fit(X_train, y_train_clasificacion)
train_accuracy = pipeline_clasificacion.score(X_train, y_train_clasificacion)
print("Precisión en el conjunto de entrenamiento:", train_accuracy)
test_accuracy = pipeline_clasificacion.score(X_test, y_test_clasificacion)
print("Precisión en el conjunto de prueba:", test_accuracy)


# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(['RainTomorrow',  'RainfallTomorrow'], axis=1)
y_regression = data['RainfallTomorrow']
y_classification = data['RainTomorrow']
X_train, X_test, y_train_regression, y_test_regression, y_train_classification, y_test_classification = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=42)

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(exclude=['number']).columns

# Crear un transformer para imputar variables numéricas con la media
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Crear un transformer para imputar variables categóricas con la moda
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combinar los transformers en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear el pipeline completo con el preprocesamiento y el modelo de regresión lineal
pipeline_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Entrenar el pipeline de regresión
pipeline_regression.fit(X_train, y_train_regression)

# Calcular el coeficiente de determinación (R^2)
y_pred_regression = pipeline_regression.predict(X_test)
r2 = r2_score(y_test_regression, y_pred_regression)
print(f"Coeficiente de determinación (R^2) del modelo de regresión: {r2}")

# Crear el pipeline completo con el preprocesamiento y el modelo de regresión logística
pipeline_classification = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Entrenar el pipeline de clasificación
pipeline_classification.fit(X_train, y_train_classification)

# Calcular el puntaje F1
y_pred_classification = pipeline_classification.predict(X_test)
f1 = f1_score(y_test_classification, y_pred_classification)
print(f"Puntaje F1 del modelo de clasificación: {f1}")

# Guardar los modelos en archivos
joblib.dump(pipeline_regression, 'weather_regression.joblib')
joblib.dump(pipeline_classification, 'weather_classification.joblib')
"""
