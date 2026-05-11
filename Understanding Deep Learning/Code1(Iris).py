import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
This code implements a simple neural network using TensorFlow to classify the Iris dataset.
The Iris dataset consists of 150 samples of iris flowers, with 4 features (sepal length, 
sepal width, petal length, petal width) and 3 classes (Iris-Setosa, Iris-Versicolor, 
Iris-Virginica). In this implementation, we will convert the problem into a 
binary classification task by classifying whether a sample is Iris-Setosa (class 0) or not 
(class 1). The model is trained on the training set and evaluated on the test set, 
with the final loss and accuracy printed at the end. 

Note: This is a simple example for educational purposes and may not achieve high accuracy due
to the binary classification approach and the simplicity of the model.
"""


# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Encode the target Labels to binary classes: 0 for Iris-Setosa, 1 for others
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Binary classification: 0 (Setosa) remains 0, 1 (Versicolor) and 2 (Virginica) become 1
y = np.where(y == 0, 0, 1)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

#Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')