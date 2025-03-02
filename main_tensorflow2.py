import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, 
    Permute, Flatten, RepeatVector, Lambda, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

# Attention Mechanism
def attention_3d_block(inputs, single_attention_vector=False):
    time_steps = tf.keras.backend.int_shape(inputs)[1]
    input_dim = tf.keras.backend.int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)

    if single_attention_vector:
        a = GlobalAveragePooling1D()(a)  
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])

    return output_attention_mul

# Create dataset function
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    
    return np.array(dataX), np.array(dataY)

# Normalize function
def NormalizeMult(data):
    data = np.array(data)
    normalize = np.zeros((data.shape[1], 2))

    for i in range(data.shape[1]):
        listlow, listhigh = np.percentile(data[:, i], [0, 100])
        normalize[i, 0] = listlow
        normalize[i, 1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            data[:, i] = (data[:, i] - listlow) / delta

    return data, normalize

# Reverse normalization function
def FNormalizeMult(data, normalize):
    data = np.array(data)
    for i in range(data.shape[1]):
        listlow, listhigh = normalize[i, 0], normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            data[:, i] = data[:, i] * delta + listlow

    return data

# Model function
def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)

    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)

    return model

# Load data
data = pd.read_csv("./pollution.csv")
data = data.drop(['date', 'wnd_dir'], axis=1)

INPUT_DIMS = 7
TIME_STEPS = 20
lstm_units = 64

# Normalize data
data, normalize = NormalizeMult(data)
pollution_data = data[:, 0].reshape(len(data), 1)

train_X, _ = create_dataset(data, TIME_STEPS)
_, train_Y = create_dataset(pollution_data, TIME_STEPS)

# Train model
model = attention_model()
model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_Y, epochs=10, batch_size=64, validation_split=0.1)


from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})

# Generate predictions using the loaded model
train_predictions = model.predict(train_X) 

train_Y = train_Y.flatten()  # Or train_Y = train_Y.reshape(-1) 
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': train_Y})


import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][:500])
plt.plot(train_results['Actuals'][:500])




