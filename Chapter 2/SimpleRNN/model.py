import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN , Dense
from sklearn.model_selection import train_test_split

# 1. Generate time series synthetic datax using sine wave with random noise
t = np.array(range(2000))
x = np.sin(0.02 * t) + 0.5 * np.random.rand(2000)*2

#train_data, test_data = x[0:1300], x[1300:]
train_data, test_data = train_test_split(x, test_size=0.3, random_state=42) #consistent results

# 2. Define converting function to create sequences
def convertToDataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

train_steps = 15
test_steps = 25
X_train, y_train = convertToDataset(train_data, train_steps)
X_test, y_test = convertToDataset(test_data, test_steps)

trainX = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3. Build RNN model
def build_model(seq_length):
    model = Sequential()
    model.add(SimpleRNN(units=64, activation='tanh', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model(train_steps)
model.summary()

# 4. Train the model
model.compile(optimizer='adam', loss='mse')

checkpoint_dir = './model_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}.weights.h5')

checkpoint_callback = ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

# 5. model fitting
history = model.fit(trainX, y_train, epochs=150, batch_size=32, verbose=2, callbacks=[checkpoint_callback])

# 6. Building new model for testing and loading the best weights
model_test = build_model(test_steps)
model_test.load_weights('./model_checkpoints/ckpt_150.weights.h5')
model_test.compile(optimizer='adam', loss='mse')

# 7. Evaluate the model on test data
test_loss = model_test.evaluate(testX, y_test, verbose=0)
print(f"Test Loss: {test_loss}")

# 8. Predict and visualize results
predicted_output = model_test.predict(testX)
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Values')
plt.plot(predicted_output, label='Predicted Values')
plt.title('True vs Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()