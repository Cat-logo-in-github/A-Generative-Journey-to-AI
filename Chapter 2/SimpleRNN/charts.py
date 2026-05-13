"""
rnn_concepts_visualizer.py

This script teaches the concepts behind the RNN example.

Concepts demonstrated:
1. Time-series windows
2. Sequential learning
3. Hidden states
4. Checkpoint loading
5. Learning progression across epochs
6. Variable-length sequence processing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

np.random.seed(42)

t = np.arange(2000)

x = np.sin(0.02 * t) + 0.5 * np.random.rand(2000) * 2


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(len(x) * 0.7)

train_data = x[:split_index]
test_data = x[split_index:]


# ============================================================
# CREATE SEQUENCES
# ============================================================

def convert_to_dataset(data, window_size):

    X = []
    y = []

    for i in range(len(data) - window_size):

        X.append(data[i:i + window_size])

        y.append(data[i + window_size])

    return np.array(X), np.array(y)


# Professor's concept:
# train on shorter sequence
# test on longer sequence

train_steps = 15
test_steps = 25

X_train, y_train = convert_to_dataset(train_data, train_steps)
X_test, y_test = convert_to_dataset(test_data, test_steps)

trainX = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
testX = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(seq_length):

    model = Sequential()

    model.add(
        SimpleRNN(
            units=64,
            activation='tanh',
            input_shape=(seq_length, 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# ============================================================
# VISUALIZE ORIGINAL SIGNAL
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(x, color='royalblue')

plt.title("Synthetic Time Series")
plt.xlabel("Time")
plt.ylabel("Signal")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# VISUALIZE TRAINING WINDOW
# ============================================================

sample_window = X_train[0]
sample_target = y_train[0]

plt.figure(figsize=(12, 5))

plt.plot(
    range(train_steps),
    sample_window,
    marker='o',
    linewidth=2,
    label='Input Sequence'
)

plt.scatter(
    train_steps,
    sample_target,
    color='red',
    s=120,
    label='Target Value'
)

plt.title("What the RNN Sees During Training")
plt.xlabel("Sequence Position")
plt.ylabel("Value")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# EXPLAIN HIDDEN STATE CONCEPT
# ============================================================

print("\\n")
print("=" * 60)
print("RNN HIDDEN STATE EXPLANATION")
print("=" * 60)

for i in range(5):

    print(f"""
Timestep {i + 1}

Input:
{sample_window[i]:.4f}

Conceptually:
new_hidden_state = f(current_input, previous_hidden_state)
""")


# ============================================================
# LOAD DIFFERENT EPOCHS
# ============================================================

checkpoint_dir = "./model_checkpoints"

epochs_to_visualize = [1, 5, 20, 50, 100, 150]

plt.figure(figsize=(15, 10))

plot_index = 1

for epoch in epochs_to_visualize:

    weight_path = os.path.join(
        checkpoint_dir,
        f"ckpt_{epoch}.weights.h5"
    )

    if not os.path.exists(weight_path):

        print(f"Missing checkpoint: {weight_path}")
        continue

    # Build test model using LONGER sequences
    model = build_model(test_steps)

    model.load_weights(weight_path)

    predictions = model.predict(testX, verbose=0)

    plt.subplot(3, 2, plot_index)

    plt.plot(
        y_test[:200],
        label='True',
        linewidth=2
    )

    plt.plot(
        predictions[:200],
        label='Predicted',
        linewidth=2
    )

    plt.title(f"Epoch {epoch}")

    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.legend()
    plt.grid(True)

    plot_index += 1

plt.tight_layout()
plt.show()


# ============================================================
# FINAL MODEL ANALYSIS
# ============================================================

final_model = build_model(test_steps)

final_model.load_weights(
    os.path.join(
        checkpoint_dir,
        "ckpt_150.weights.h5"
    )
)

predictions = final_model.predict(testX, verbose=0)

errors = np.abs(
    y_test - predictions.flatten()
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 10)
)

# ------------------------------------------------
# TRUE VS PREDICTED
# ------------------------------------------------

axes[0].plot(
    y_test[:300],
    label='True Values',
    linewidth=2
)

axes[0].plot(
    predictions[:300],
    label='Predicted Values',
    linewidth=2
)

axes[0].set_title("Final RNN Predictions")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Signal")

axes[0].legend()
axes[0].grid(True)

# ------------------------------------------------
# ERROR GRAPH
# ------------------------------------------------

axes[1].plot(
    errors[:300],
    color='crimson'
)

axes[1].set_title("Prediction Error")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Absolute Error")

axes[1].grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# EDUCATIONAL SUMMARY
# ============================================================

print("\\n")
print("=" * 60)
print("KEY RNN CONCEPTS")
print("=" * 60)

print("""

1. SEQUENCE LEARNING
--------------------
RNNs learn from ordered temporal data.

2. SHARED RECURRENT WEIGHTS
---------------------------
The same recurrent cell is reused across time.

3. HIDDEN STATE MEMORY
----------------------
The hidden state carries temporal information.

4. VARIABLE-LENGTH PROCESSING
-----------------------------
An RNN can often process longer sequences than
those used during training.

5. CHECKPOINTING
----------------
Checkpoint files allow:
- restoring models
- comparing epochs
- continuing training

6. FORECASTING
--------------
The network learns:
"Use past values to predict future values."

""")

print("=" * 60)