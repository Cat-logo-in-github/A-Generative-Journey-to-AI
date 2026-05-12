# What Model does:
---

In IMDb:

* `X_train` = movie reviews converted into numbers
* `y_train` = sentiment labels

  * `0` → negative
  * `1` → positive

The model learns:

> “Given this review text, predict whether sentiment is positive or negative.”

---

# Code Pipeline

1. Load labeled movie reviews
2. Convert reviews to equal-length sequences
3. Build an LSTM neural network
4. Train on known examples
5. Predict sentiment for new text

---

# Step-by-Step Explanation

---

# 1. Import Libraries

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

### What this does

* `tensorflow` → deep learning framework
* `imdb` → built-in movie review dataset
* `pad_sequences` → makes all reviews same length

---

# 2. Setup Parameters

```python
vocab_size = 10000
max_sequence_length = 200
```

---

## `vocab_size = 10000`

Use only the **10,000 most common words**.

Example:

| Word  | Index |
| ----- | ----- |
| the   | 1     |
| movie | 14    |
| great | 88    |

Rare words are ignored.

This reduces complexity.

---

## `max_sequence_length = 200`

Every review becomes length 200.

Why?

Neural networks need equal-sized input.

Some reviews are:

* 50 words
* 300 words
* 120 words

So we standardize them.

---

# 3. Load Dataset

```python
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
```

This is the line that splits training and testing data.

---

# THIS is the train/test split

The dataset already comes pre-split.

So after this line:

| Variable  | Meaning          |
| --------- | ---------------- |
| `X_train` | training reviews |
| `y_train` | training labels  |
| `X_test`  | testing reviews  |
| `y_test`  | testing labels   |

---

## What does one review look like?

Instead of text:

```text
"This movie was great"
```

You get:

```python
[1, 14, 20, 88]
```

These are word IDs.

---

# Why supervised learning?

Because labels exist.

Example:

| Review            | Label |
| ----------------- | ----- |
| "Amazing movie"   | 1     |
| "Terrible acting" | 0     |

The model compares:

* prediction
* correct answer

Then adjusts weights.

That is supervised learning.

---

# 4. Pad Sequences

```python
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)
```

---

## Why needed?

Neural networks require fixed-size inputs.

Example before padding:

```python
[1, 5, 7]
[1, 8, 2, 9, 3, 4]
```

After padding to length 6:

```python
[0, 0, 0, 1, 5, 7]
[1, 8, 2, 9, 3, 4]
```

Now all reviews have same length.

---

# 5. Build the Model

```python
model = tf.keras.Sequential([
```

This creates a layer-by-layer neural network.

---

# Layer 1 — Embedding Layer

```python
tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=128,
    input_length=max_sequence_length
)
```

This is VERY important.

---

## What is Embedding?

Words cannot directly go into neural networks.

The embedding layer converts word IDs into vectors.

Example:

```python
"great" → [0.12, -0.44, 0.91, ...]
```

Each word becomes a 128-dimensional vector.

---

## Parameters

### `input_dim=vocab_size`

Vocabulary size = 10,000 words.

---

### `output_dim=128`

Each word becomes a vector of length 128.

---

### `input_length=max_sequence_length`

Each review has 200 words.

So input shape becomes:

```python
(200 words × 128 features)
```

---

# Layer 2 — LSTM

```python
tf.keras.layers.LSTM(
    64,
    dropout=0.2,
    recurrent_dropout=0.2
)
```

This replaces the simple dense layers from your earlier 8-neuron network.

---

# What is LSTM?

LSTM = Long Short-Term Memory network.

It is designed for sequences/text.

Unlike a normal neural network:

* it reads words one by one
* remembers previous context

Example:

```text
"The movie was not good"
```

The word `"not"` changes meaning later.

LSTM can remember that.

---

## Why not regular Dense layers?

Dense layers:

* treat all inputs independently

LSTM:

* understands order and context

Text requires order understanding.

---

# What does `64` mean?

```python
LSTM(64)
```

64 memory units (neurons).

These learn language patterns.

---

# What is dropout?

```python
dropout=0.2
```

Randomly disables 20% neurons during training.

Prevents overfitting.

---

# What is recurrent_dropout?

```python
recurrent_dropout=0.2
```

Drops some recurrent connections inside the LSTM memory system.

Again helps generalization.

---

# Layer 3 — Output Layer

```python
tf.keras.layers.Dense(1, activation='sigmoid')
```

---

# Why only 1 neuron?

Binary classification:

* positive
* negative

One probability output is enough.

---

# Why sigmoid instead of ReLU?

Because sigmoid outputs values between 0 and 1.

Perfect for probability.

Example:

```python
0.93 → positive
0.12 → negative
```

---

# Why NO ReLU?

Excellent question.

---

## ReLU is usually used in hidden layers

Example:

```python
Dense(64, activation='relu')
```

But here:

* the hidden processing is done by the LSTM
* LSTM already has its own internal activations:

  * sigmoid
  * tanh

So you don't manually add ReLU.

---

# 6. Compile Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

# Optimizer

```python
optimizer='adam'
```

Controls weight updates.

Adam is popular and efficient.

---

# Loss Function

```python
loss='binary_crossentropy'
```

Used for binary classification.

Measures prediction error.

---

# Metric

```python
metrics=['accuracy']
```

Shows percentage correct.

---

# 7. Train Model

```python
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)
```

---

# What happens during training?

The model:

1. Takes reviews
2. Predicts sentiment
3. Compares with actual labels
4. Adjusts weights
5. Repeats many times

---

# `epochs=5`

Entire dataset processed 5 times.

---

# `batch_size=32`

Train on 32 reviews at once before updating weights.

---

# `validation_split=0.1`

Uses:

* 90% of training data for training
* 10% for validation

Validation checks performance during training.

---

# Important distinction

You have BOTH:

* test set (`X_test`)
* validation split

---

## Validation Set

Used DURING training.

---

## Test Set

Used AFTER training for final evaluation.

Example:

```python
model.evaluate(X_test, y_test)
```

---

# 8. Prediction Function

```python
def predict_sentiment(review_text):
```

Takes custom text input.

---

# Get Word Dictionary

```python
word_to_index = imdb.get_word_index()
```

Maps words → numbers.

---

# Convert words to tokens

```python
tokens = [word_to_index.get(word, -3) + 3 for word in review_text.lower().split()]
```

Example:

```python
"great movie"
```

becomes:

```python
[88, 14]
```

---

# Why `+3`?

IMDb reserves:

* 0 = padding
* 1 = start token
* 2 = unknown
* 3 = unused offset

So actual words begin at index 3.

---

# Unknown words

```python
word_to_index.get(word, -3)
```

If word missing:

* returns `-3`
* after `+3` becomes `0`

---

# Keep within vocabulary

```python
tokens = [t if t < vocab_size else 2 for t in tokens]
```

Words outside top 10,000 become:

* `2` = unknown token

---

# Pad Input

```python
padded_tokens = pad_sequences([tokens], maxlen=max_sequence_length)
```

Makes input length 200.

---

# Predict

```python
prediction = model.predict(padded_tokens)[0][0]
```

Example output:

```python
0.91
```

Meaning:

* 91% positive probability

---

# Final Decision

```python
return "Positive" if prediction > 0.5 else "Negative"
```

Threshold:

* > 0.5 → positive
* ≤ 0.5 → negative

---

# Main Concept Difference From Your Earlier 8-Neuron Network

Your previous network probably looked like:

```python
Dense(8, activation='relu')
```

That works for:

* tabular data
* fixed features

But text is sequential.

So here:

* Embedding learns word meaning
* LSTM learns word order/context
* Sigmoid predicts probability

---

# Architecture Summary

```text
Text
 ↓
Word IDs
 ↓
Embedding Layer
 ↓
LSTM Memory Network
 ↓
Sigmoid Output
 ↓
Positive / Negative
```

---

# One More Important Thing

This model is much more advanced than a simple feedforward neural network because it learns:

* semantic meaning
* sequence relationships
* contextual dependencies

That is why NLP commonly uses:

* LSTM
* GRU
* Transformers

instead of basic Dense networks.


# Model Training Performance:

> Epoch 1/5
> 704/704 ━━━━━━━━━━━━━━━━━━━━ 98s 137ms/step - accuracy: 0.7708 - loss: 0.4813 - val_accuracy: 0.8464 - val_loss: 0.3716
> Epoch 2/5
> 704/704 ━━━━━━━━━━━━━━━━━━━━ 127s 116ms/step - accuracy: 0.8625 - loss: 0.3352 - val_accuracy: 0.7112 - val_loss: 0.7464
> Epoch 3/5
> 704/704 ━━━━━━━━━━━━━━━━━━━━ 83s 117ms/step - accuracy: 0.8655 - loss: 0.3272 - val_accuracy: 0.8420 - val_loss: 0.4512
> Epoch 4/5
> 704/704 ━━━━━━━━━━━━━━━━━━━━ 88s 126ms/step - accuracy: 0.9058 - loss: 0.2443 - val_accuracy: 0.8704 - val_loss: 0.3470
> Epoch 5/5
> 704/704 ━━━━━━━━━━━━━━━━━━━━ 79s 113ms/step - accuracy: 0.9219 - loss: 0.2031 - val_accuracy: 0.8592 - val_loss: 0.3721

### Model Stored As: sentiment_model.keras

# Prediction Run:

> 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 257ms/step

> Review: This movie was fantastic I really enjoyed it
> Predicted Sentiment: Positive