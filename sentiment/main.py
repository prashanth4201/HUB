import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import gym
import gym.spaces
import random
from collections import deque
from flask import Flask, request, jsonify


# Load and preprocess METR-LA dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.iloc[:, 1:])  # Normalize features
    return df_scaled, scaler


data, scaler = load_data("metr-la.csv")


# Prepare time-series data for LSTM
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])  # Predicting next time step of first sensor
    return np.array(X), np.array(y)


X, y = create_sequences(data)
X_train, X_test, y_train, y_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):], y[:int(0.8 * len(y))], y[int(0.8 * len(
    y)):]

# LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


def predict_traffic(input_data):
    scaled_input = scaler.transform([input_data])
    seq = np.array([scaled_input[-10:]])  # Ensure correct sequence length
    prediction = model.predict(seq)
    return scaler.inverse_transform([[prediction[0][0]]])[0][0]


# Reinforcement Learning for Traffic Light Control
class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 0: Short Red, 1: Balanced, 2: Short Green
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = np.zeros(10)

    def step(self, action):
        congestion_level = predict_traffic(self.state)
        reward = -congestion_level if action == 2 else congestion_level  # Encourage lower congestion
        self.state = np.roll(self.state, -1)
        self.state[-1] = congestion_level
        return self.state, reward, False, {}

    def reset(self):
        self.state = np.zeros(10)
        return self.state


env = TrafficEnv()


def train_rl_agent():
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = Sequential([
        Dense(24, activation='relu', input_dim=state_size),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


dqn_model = train_rl_agent()

# Flask API for Traffic Prediction and Light Optimization
app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Real-Time Traffic Prediction API! Use /predict or /optimize."


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = predict_traffic(data)
    return jsonify({'predicted_traffic': prediction})


@app.route('/optimize', methods=['POST'])
def optimize():
    state = np.array(request.json['state'])
    action = np.argmax(dqn_model.predict(np.array([state]))[0])
    return jsonify({'recommended_action': int(action)})


if __name__ == '__main__':
    app.run(debug=True)