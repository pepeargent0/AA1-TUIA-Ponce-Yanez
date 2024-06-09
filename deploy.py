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
from scikeras.wrappers import KerasRegressor


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
        self.train = self.df[self.df['Date'] <= date_train_limit].copy()
        self.test = self.df[self.df['Date'] > date_train_limit].copy()

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
        self.train = self.train.drop(columns=['Date']).copy()
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

def show_metrics_regresion(y_true, y_pred, mensaje, verbose=True):
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

def create_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(12,), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

modelo_lluvia = ModeloLluvia('weatherAUS.csv')
(y_train_regresion, y_test_regresion, y_train_clasificacion, y_test_clasificacion, X_train,X_test)=modelo_lluvia.crear()
keras_regressor = KerasRegressor(model=create_model, epochs=150, batch_size=16, verbose=0)
pipeline = Pipeline(steps=[
    ('regressor', keras_regressor)
])
pipeline.fit(X_train, y_train_regresion)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
show_metrics_regresion(y_train_regresion, y_pred_train, "Métricas del conjunto de entrenamiento:", True)
show_metrics_regresion(y_test_regresion, y_pred_test, "Métricas del conjunto de prueba:", True)