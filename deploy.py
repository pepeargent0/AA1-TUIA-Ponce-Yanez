import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, f1_score
from scikeras.wrappers import KerasRegressor








import joblib


class ModeloLluvia:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, sep=',')
        self.train = None
        self.test = None
        self.y_train_regresion = None
        self.y_test_regresion = None
        self.y_train_clasificacion = None
        self.y_test_clasificacion = None
        self.X_train = None
        self.X_test = None

    def _seprar_test_train(self):
        ciudades = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne',
                    'MelbourneAirport', 'MountGambier',
                    'Sydney', 'SydneyAirport']
        self.data = self.data[self.data['Location'].isin(ciudades)]
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values(by='Date')
        date_train_limit = pd.to_datetime('2015-10-06')
        self.train = self.data[self.data['Date'] <= date_train_limit]
        self.test = self.data[self.data['Date'] > date_train_limit]

        self.train = self.train.drop(columns=['Unnamed: 0'])
        self.train = self.train.drop(columns=['Location'])
        self.test = self.test.drop(columns=['Unnamed: 0'])
        self.test = self.test.drop(columns=['Location'])
        self.y_train_regresion = self.train['RainfallTomorrow']
        self.y_train_clasificacion = self.train['RainTomorrow']
        self.y_test_regresion = self.test['RainfallTomorrow']
        self.y_test_clasificacion = self.test['RainTomorrow']

    def _procesar_train(self):
        columns_to_aggregate = ['Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Humidity9am',
                                'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'WindSpeed3pm', 'WindSpeed9am']

        self.train['PressureVariation'] = self.train['Pressure3pm'] - self.train['Pressure9am']
        self.train['TempVariation'] = self.train['Temp3pm'] - self.train['Temp9am']
        self.train['HumidityVariation'] = self.train['Humidity3pm'] - self.train['Humidity9am']
        self.train['CloudVariation'] = self.train['Cloud3pm'] - self.train['Cloud9am']
        self.train['WindSpeedVariation'] = self.train['WindSpeed3pm'] - self.train['WindSpeed9am']
        self.train.drop(columns=columns_to_aggregate, inplace=True)
        self.train['RainToday'] = self.train['RainToday'].map({'No': 0, 'Yes': 1})
        self.train['RainTomorrow'] = self.train['RainTomorrow'].map({'No': 0, 'Yes': 1})
        self.train = self.train.drop(['RainfallTomorrow', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainTomorrow'],
                                       axis=1)
        self.train.fillna(method='ffill', inplace=True)


    def _procesar_test(self):
        self.test.sort_values(by='Date', inplace=True)
        self.test.fillna(method='ffill', inplace=True)
        for column in self.test.columns:
            self.test[column] = self.test[column].ffill()
            self.test[column] = self.test[column].bfill()

        columns_to_aggregate = ['Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Humidity9am',
                                'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'WindSpeed3pm', 'WindSpeed9am']
        self.test['PressureVariation'] = self.test['Pressure3pm'] - self.test['Pressure9am']
        self.test['TempVariation'] = self.test['Temp3pm'] - self.test['Temp9am']
        self.test['HumidityVariation'] = self.test['Humidity3pm'] - self.test['Humidity9am']
        self.test['CloudVariation'] = self.test['Cloud3pm'] - self.test['Cloud9am']
        self.test['WindSpeedVariation'] = self.test['WindSpeed3pm'] - self.test['WindSpeed9am']
        self.test['RainToday'] = self.test['RainToday'].map({'No': 0, 'Yes': 1})
        self.test['RainTomorrow'] = self.test['RainTomorrow'].map({'No': 0, 'Yes': 1})
        self.test.drop(columns=columns_to_aggregate, inplace=True)
        self.test = self.test.drop(['RainfallTomorrow', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainTomorrow'],
                                     axis=1)


    def crear(self):
        self._seprar_test_train()
        self._procesar_train()
        self._procesar_test()
        self.y_test_regresion.ffill(inplace=True)
        self.y_train_regresion.ffill(inplace=True)
        self.test = self.test.drop(columns=['Date'])
        self.train = self.train.drop(columns=['Date'])
        return (self.y_train_regresion, self.y_test_regresion, self.y_train_clasificacion, self.y_test_clasificacion,
                self.train, self.test)


modelo_lluvia = ModeloLluvia('weatherAUS.csv')
(y_train_regresion, y_test_regresion, y_train_clasificacion, y_test_clasificacion,
 X_train, X_test) = modelo_lluvia.crear()

def crear_modelo_regresion(num_hidden_layers=3, hidden_layer_size=23):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=12, activation='relu'))
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(hidden_layer_size, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def show_r2_score(y_true, y_pred, message):
    r2 = r2_score(y_true, y_pred)
    print(message)
    print("R^2 Score:", r2)


# Usar la clase para procesar el dataset
modelo_lluvia = ModeloLluvia('weatherAUS.csv')
(y_train_regresion, y_test_regresion, y_train_clasificacion, y_test_clasificacion,
 X_train, X_test) = modelo_lluvia.crear()

# Crear el pipeline
pipeline_regresion = Pipeline([
    ('scaler', RobustScaler()),
    ('regressor', KerasRegressor(build_fn=crear_modelo_regresion, epochs=200, batch_size=32, validation_split=0.2, verbose=1))
])

pipeline_regresion.fit(X_train, y_train_regresion)
y_pred_train_regresion = pipeline_regresion.predict(X_train)
y_pred_test_regresion = pipeline_regresion.predict(X_test)
show_r2_score(y_train_regresion, y_pred_train_regresion, "Métricas del conjunto de entrenamiento:")
show_r2_score(y_test_regresion, y_pred_test_regresion, "Métricas del conjunto de prueba:")

"""
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
