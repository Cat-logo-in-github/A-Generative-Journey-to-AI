Yes — this is a classic example of training an **MLP (Multilayer Perceptron)**.

Specifically, your code builds:

```text
Input Layer
   ↓
Hidden Dense Layer (128 neurons + ReLU)
   ↓
Output Dense Layer (10 neurons + Softmax)
```

That architecture is exactly a:

# ✅ Multilayer Perceptron (MLP)

---

# Why it is called an MLP

## 1. “Perceptron”

A perceptron is a simple neuron:

```text
output = activation(weights · inputs + bias)
```

Your `Dense` layers are made of many perceptrons.

---

## 2. “Multi-layer”

You have more than one layer of neurons:

* hidden layer
* output layer

So:

```text
single perceptron → not MLP
multiple dense layers → MLP
```

---

# Your exact architecture

This line:

```python
model.add(Flatten(input_shape=(28, 28)))
```

converts:

```text
28×28 image
```

into:

```text
784-dimensional vector
```

Then:

```python
Dense(128)
```

means:

```text
784 → 128 fully connected neurons
```

Then:

```python
Dense(10)
```

means:

```text
128 → 10 output neurons
```

(one per digit class)

---

# Why this is NOT a CNN

Many beginners confuse this.

Your model does NOT use:

* convolutions
* kernels
* feature maps
* pooling

So it is NOT:

* CNN
* ResNet
* ConvNet

It is purely:

### fully connected layers

which defines an MLP.

---

# Visualization of your network

```text
MNIST Image (28×28)
        ↓
Flatten
        ↓
[784 inputs]
        ↓
Dense(128) + ReLU
        ↓
Dense(10) + Softmax
        ↓
Digit probabilities
```

---

# What the network learns

During training:

```python
model.fit(...)
```

the MLP adjusts:

* weights
* biases

using:

* backpropagation
* gradient descent (Adam optimizer)

to reduce:

```python
loss='sparse_categorical_crossentropy'
```

---

# Historically important fact

MNIST + MLPs were one of the earliest major deep learning demos before CNNs became dominant.

An MLP on MNIST usually gets:

* ~97–98% accuracy

CNNs later pushed this:

* > 99%

because they understand spatial structure better.

---

# Technically speaking

Your model is:

### ✅ Feedforward Neural Network (FNN)

### ✅ Dense Neural Network

### ✅ Fully Connected Network

### ✅ MLP

These terms are closely related here.

---

# One subtle point

Some strict definitions say:

> An MLP should have at least one hidden layer.

You do have one:

```python
Dense(128)
```

So yours fully qualifies.


# Output:

> Starting training...
> Epoch 1/5
> 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9275 - loss: 0.2560     
> Epoch 2/5
> 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9660 - loss: 0.1139  
> Epoch 3/5
> 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9758 - loss: 0.0786  
> Epoch 4/5
> 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9819 - loss: 0.0591  
> Epoch 5/5
> 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9858 - loss: 0.0452  

> Evaluating on test data...
> 313/313 - 1s - 2ms/step - accuracy: 0.9787 - loss: 0.0767

> Final Test Accuracy: 97.87%

> Generating predictions for the first 5 test images...
> 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step