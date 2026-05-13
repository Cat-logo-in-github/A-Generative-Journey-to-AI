import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import cifar10

# ============================================================
# LOAD MODEL
# ============================================================

model = load_model("cifar10_cnn_model.keras")

# IMPORTANT FIX FOR NEWER KERAS
dummy_input = np.zeros((1, 32, 32, 3), dtype=np.float32)
model(dummy_input)

# ============================================================
# LOAD DATA
# ============================================================

(_, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype("float32") / 255.0

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ============================================================
# SELECT SAMPLE IMAGE
# ============================================================

image_index = 0

sample_image = x_test[image_index]
sample_label = y_test[image_index][0]

# ============================================================
# SHOW ORIGINAL IMAGE
# ============================================================

plt.figure(figsize=(4, 4))

plt.imshow(sample_image)

plt.title(f"Original Image\nLabel: {class_names[sample_label]}")

plt.axis("off")

plt.tight_layout()

plt.savefig("01_original_image.png")

plt.show()

# ============================================================
# CREATE FEATURE EXTRACTION MODEL
# ============================================================

layer_outputs = []
layer_names = []

for layer in model.layers:

    if "conv" in layer.name or "dense" in layer.name:

        layer_outputs.append(layer.output)

        layer_names.append(layer.name)

activation_model = Model(
    inputs=model.inputs,
    outputs=layer_outputs
)

# ============================================================
# RUN IMAGE THROUGH NETWORK
# ============================================================

input_image = np.expand_dims(sample_image, axis=0)

activations = activation_model.predict(input_image)

# ============================================================
# VISUALIZE CONVOLUTION FILTERS (KERNELS)
# ============================================================

conv_layer_count = 0

for layer in model.layers:

    if "conv" in layer.name:

        conv_layer_count += 1

        filters, biases = layer.get_weights()

        print("\n================================================")
        print(f"CONVOLUTION LAYER {conv_layer_count}")
        print("================================================")

        print("Layer name:", layer.name)
        print("Filter tensor shape:", filters.shape)

        print("""
Filter Tensor Meaning:
(kernel_height, kernel_width, input_channels, number_of_filters)
""")

        plt.figure(figsize=(8, 8))

        for i in range(9):

            plt.subplot(3, 3, i + 1)

            # Show first input channel
            f = filters[:, :, 0, i]

            plt.imshow(f, cmap='viridis')

            plt.title(f"Filter {i}")

            plt.axis('off')

        plt.suptitle(
            f"Conv Layer {conv_layer_count} Learned Filters",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(f"02_conv{conv_layer_count}_filters.png")

        plt.show()

# ============================================================
# VISUALIZE FEATURE MAPS
# ============================================================

conv_activation_index = 0

for i, layer in enumerate(model.layers):

    if "conv" in layer.name:

        feature_maps = activations[conv_activation_index]

        conv_activation_index += 1

        print("\n================================================")
        print(f"FEATURE MAPS FOR {layer.name}")
        print("================================================")

        print("Feature map shape:", feature_maps.shape)

        print("""
Feature Map Shape Meaning:
(batch_size, height, width, channels)

Each channel is one learned representation of the image.
""")

        plt.figure(figsize=(10, 10))

        for j in range(9):

            plt.subplot(3, 3, j + 1)

            plt.imshow(
                feature_maps[0, :, :, j],
                cmap='viridis'
            )

            plt.title(f"Map {j}")

            plt.axis('off')

        plt.suptitle(
            f"Feature Maps - {layer.name}",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(f"03_{layer.name}_feature_maps.png")

        plt.show()

# ============================================================
# VISUALIZE DENSE LAYER ACTIVATIONS
# ============================================================

dense_layer_counter = 0

for idx, layer in enumerate(model.layers):

    if "dense" in layer.name:

        dense_layer_counter += 1

        dense_output = activations[idx][0]

        print("\n================================================")
        print(f"DENSE LAYER {dense_layer_counter}")
        print("================================================")

        print("Activation shape:", dense_output.shape)

        print("""
Dense neurons combine abstract information
learned from convolution layers.
""")

        # Make square-ish heatmap
        padded = np.zeros(64)

        length = min(len(dense_output), 64)

        padded[:length] = dense_output[:length]

        heatmap = padded.reshape(8, 8)

        plt.figure(figsize=(6, 6))

        plt.imshow(heatmap, cmap='hot')

        plt.colorbar()

        plt.title(
            f"Dense Layer {dense_layer_counter} Activations"
        )

        plt.axis('off')

        plt.tight_layout()

        plt.savefig(f"04_dense_{dense_layer_counter}.png")

        plt.show()

# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

predictions = model.predict(input_image)

probabilities = tf.nn.softmax(predictions[0]).numpy()

predicted_class = np.argmax(probabilities)

print("\n================================================")
print("FINAL PREDICTION")
print("================================================")

print("Predicted Class:", class_names[predicted_class])

plt.figure(figsize=(10, 4))

bars = plt.bar(class_names, probabilities)

bars[predicted_class].set_color('red')

plt.xticks(rotation=45)

plt.ylabel("Probability")

plt.title(
    f"Prediction Probabilities\n"
    f"Predicted: {class_names[predicted_class]}"
)

plt.tight_layout()

plt.savefig("05_prediction_probabilities.png")

plt.show()

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n================================================")
print("MODEL SUMMARY")
print("================================================\n")

model.summary()

# ============================================================
# SHAPE TRANSFORMATIONS THROUGH NETWORK
# ============================================================

print("\n================================================")
print("SHAPE FLOW THROUGH NETWORK")
print("================================================\n")

print("Input shape:", input_image.shape)

for layer in model.layers:

    intermediate_model = Model(
        inputs=model.inputs,
        outputs=layer.output
    )

    output = intermediate_model.predict(input_image)

    print(f"{layer.name:20s} -> {output.shape}")

print("\n================================================")
print("VISUALIZATION COMPLETE")
print("================================================")

print("""
Generated Files:

01_original_image.png
02_conv1_filters.png
02_conv2_filters.png
02_conv3_filters.png

03_conv2d_feature_maps.png
03_conv2d_1_feature_maps.png
03_conv2d_2_feature_maps.png

04_dense_1.png
04_dense_2.png

05_prediction_probabilities.png
""")