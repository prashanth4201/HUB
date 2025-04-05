import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
data_path = "/Users/vasanthkumar/PycharmProjects/summaintainence/train_FD001.txt"  # Adjust path accordingly
df = pd.read_csv(data_path, sep=" ", header=None, engine='python')


# Data preprocessing
def preprocess_data(df):
    df = df.dropna(axis=1, how='all')  # Drop empty columns
    columns = ['unit', 'time', 'operational_setting_1', 'operational_setting_2',
               'operational_setting_3', 'sensor_1', 'sensor_2', 'sensor_3',
               'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
               'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13',
               'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
               'sensor_19', 'sensor_20', 'sensor_21']
    df.columns = columns

    # Compute Remaining Useful Life (RUL)
    max_cycle = df.groupby('unit')['time'].max()
    df['RUL'] = df.apply(lambda row: max_cycle[row['unit']] - row['time'], axis=1)

    return df


df = preprocess_data(df)

# Feature selection
features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_11', 'sensor_13', 'sensor_15']
X = df[features]
y = (df['RUL'] <= 20).astype(int)  # Binary classification: Failure within 20 cycles

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# Train LSTM model
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

# Evaluate LSTM model
y_pred_lstm = (model.predict(X_test_lstm) > 0.5).astype(int)
print("LSTM Accuracy:", accuracy_score(y_test, y_pred_lstm))

# Save trained models
import joblib

joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(xgb, 'xgboost_model.pkl')
model.save('lstm_model.h5')