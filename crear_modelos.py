import joblib
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, log_loss, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def show_metrics_regresion(y_true, y_pred, mensaje, verbose=True):
    """Calcula y muestra las métricas de regresión."""
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


class DatasetReader(BaseEstimator, TransformerMixin):
    """Lee el dataset desde un archivo CSV y lo filtra según las ciudades especificadas."""

    def __init__(self, file_path):
        self.file_path = file_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            df = pd.read_csv(self.file_path, sep=',')
            ciudades = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier',
                        'Sydney', 'SydneyAirport']
            df = df[df['Location'].isin(ciudades)]
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            return df
        except Exception as e:
            raise RuntimeError(f"Error leyendo el archivo {self.file_path}: {e}")


class TrainTestSplitter(BaseEstimator, TransformerMixin):
    """Divide el dataset en conjuntos de entrenamiento y prueba según una fecha límite."""

    def __init__(self, date_train_limit):
        self.date_train_limit = date_train_limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        train = X[X['Date'] <= self.date_train_limit].copy()
        test = X[X['Date'] > self.date_train_limit].copy()
        return train, test


class TrainDataCleaner(BaseEstimator, TransformerMixin):
    """Limpia y transforma los datos de entrenamiento."""

    def __init__(self):
        self.columns_to_drop = ['Unnamed: 0', 'Location']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X_cleaned = X.copy()
            X_cleaned.drop(columns=self.columns_to_drop, inplace=True)
            X_cleaned.sort_values(by='Date', inplace=True)
            X_cleaned.fillna(method='ffill', inplace=True)
            X_cleaned['PressureVariation'] = X_cleaned['Pressure3pm'] - X_cleaned['Pressure9am']
            X_cleaned['TempVariation'] = X_cleaned['Temp3pm'] - X_cleaned['Temp9am']
            X_cleaned['HumidityVariation'] = X_cleaned['Humidity3pm'] - X_cleaned['Humidity9am']
            X_cleaned['CloudVariation'] = X_cleaned['Cloud3pm'] - X_cleaned['Cloud9am']
            X_cleaned['WindSpeedVariation'] = X_cleaned['WindSpeed3pm'] - X_cleaned['WindSpeed9am']
            X_cleaned['RainToday'] = X_cleaned['RainToday'].map({'No': 0, 'Yes': 1})
            X_cleaned['RainTomorrow'] = X_cleaned['RainTomorrow'].map({'No': 0, 'Yes': 1})
            X_cleaned.drop(columns=['Date'], inplace=True)
            columns_to_aggregate = ['Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Humidity9am',
                                    'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'WindSpeed3pm', 'WindSpeed9am',
                                    'WindGustDir',
                                    'WindDir9am', 'WindDir3pm']
            X_cleaned.drop(columns=columns_to_aggregate, inplace=True, axis=1)
            X_train = X_cleaned.drop(columns=['RainTomorrow', 'RainfallTomorrow'])
            y_train_regresion = X_cleaned['RainfallTomorrow']
            y_train_clasificacion = X_cleaned['RainTomorrow']
            return y_train_regresion, y_train_clasificacion, X_train
        except Exception as e:
            raise RuntimeError(f"Error limpiando y transformando los datos de entrenamiento: {e}")


class TestDataCleaner(BaseEstimator, TransformerMixin):
    """Limpia y transforma los datos de prueba."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X_copy = X.copy()
            X_copy['RainToday'] = X_copy['RainToday'].map({'No': 0, 'Yes': 1})
            X_copy['RainTomorrow'] = X_copy['RainTomorrow'].map({'No': 0, 'Yes': 1})
            for column in X_copy.columns:
                X_copy[column] = X_copy[column].ffill()
                X_copy[column] = X_copy[column].bfill()
            columns_to_drop = ['Unnamed: 0', 'Location', 'Date']
            X_copy.drop(columns=columns_to_drop, errors='ignore', inplace=True)
            X_copy['PressureVariation'] = X_copy['Pressure3pm'] - X_copy['Pressure9am']
            X_copy['TempVariation'] = X_copy['Temp3pm'] - X_copy['Temp9am']
            X_copy['HumidityVariation'] = X_copy['Humidity3pm'] - X_copy['Humidity9am']
            X_copy['CloudVariation'] = X_copy['Cloud3pm'] - X_copy['Cloud9am']
            X_copy['WindSpeedVariation'] = X_copy['WindSpeed3pm'] - X_copy['WindSpeed9am']
            columns_to_aggregate = ['Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Humidity9am',
                                    'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'WindSpeed3pm', 'WindSpeed9am',
                                    'WindGustDir',
                                    'WindDir9am', 'WindDir3pm']
            X_copy.drop(columns=columns_to_aggregate, inplace=True, axis=1)
            X_test = X_copy.drop(columns=['RainTomorrow', 'RainfallTomorrow'])
            y_test_regresion = X_copy['RainfallTomorrow']
            y_test_clasificacion = X_copy['RainTomorrow']
            return y_test_regresion, y_test_clasificacion, X_test
        except Exception as e:
            raise RuntimeError(f"Error limpiando y transformando los datos de prueba: {e}")


def main():
    try:
        # Pipeline de división de datos
        pipeline_split = Pipeline([
            ('dataset_reader', DatasetReader('weatherAUS.csv')),
            ('train_test_splitter', TrainTestSplitter(pd.to_datetime('2015-10-06'))),
        ])
        train, test = pipeline_split.fit_transform(None)

        # Pipeline de limpieza de datos de entrenamiento
        pipeline_train = Pipeline([
            ('train', TrainDataCleaner()),
        ])
        y_train_regresion, y_train_clasificacion, X_train = pipeline_train.fit_transform(train)

        # Pipeline de limpieza de datos de prueba
        pipeline_test = Pipeline([
            ('test', TestDataCleaner()),
        ])
        y_test_regresion, y_test_clasificacion, X_test = pipeline_test.fit_transform(test)

        # Modelo de regresión
        modelo_regresion = Sequential([
            Dense(32, input_shape=(12,), activation='relu'),
            Dense(1, activation='linear')
        ])
        modelo_regresion.compile(optimizer='adam', loss='mean_squared_error')
        pipeline_regression = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('regressor', modelo_regresion)
        ])
        pipeline_regression.fit(X_train, y_train_regresion, regressor__epochs=100, regressor__verbose=0)
        y_pred_train = pipeline_regression.predict(X_train)
        y_pred_test = pipeline_regression.predict(X_test)
        show_metrics_regresion(y_train_regresion, y_pred_train, "Métricas del conjunto de entrenamiento:", True)
        show_metrics_regresion(y_test_regresion, y_pred_test, "Métricas del conjunto de prueba:", True)
        joblib.dump(pipeline_regression, 'weather_regression.pkl')

        # Modelo de clasificación
        best_params = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.005958432842192192,
            'learning_rate_init': 0.0001422171191037891
        }
        pipeline_clasificacion = Pipeline([
            ('scaler', RobustScaler()),
            ('mlp', MLPClassifier(**best_params, max_iter=200))
        ])
        pipeline_clasificacion.fit(X_train, y_train_clasificacion)
        y_train_pred = pipeline_clasificacion.predict(X_train)
        train_accuracy = accuracy_score(y_train_clasificacion, y_train_pred)
        train_loss = log_loss(y_train_clasificacion, pipeline_clasificacion.predict_proba(X_train))
        train_recall = recall_score(y_train_clasificacion, y_train_pred, average='weighted')
        print("Precisión en el conjunto de entrenamiento:", train_accuracy)
        print("Pérdida en el conjunto de entrenamiento:", train_loss)
        print("Recall en el conjunto de entrenamiento:", train_recall)

        y_test_pred = pipeline_clasificacion.predict(X_test)
        test_accuracy = accuracy_score(y_test_clasificacion, y_test_pred)
        test_loss = log_loss(y_test_clasificacion, pipeline_clasificacion.predict_proba(X_test))
        test_recall = recall_score(y_test_clasificacion, y_test_pred, average='weighted')
        print("Precisión en el conjunto de prueba:", test_accuracy)
        print("Pérdida en el conjunto de prueba:", test_loss)
        print("Recall en el conjunto de prueba:", test_recall)

        joblib.dump(pipeline_clasificacion, 'weather_clasificacion.pkl')
    except Exception as e:
        print(f"Ocurrió un error: {e}")


if __name__ == "__main__":
    main()
