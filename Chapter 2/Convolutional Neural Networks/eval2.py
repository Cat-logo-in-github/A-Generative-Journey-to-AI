import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import cifar10

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "cnn_visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

model = load_model("cifar10_cnn_model.keras")

# Fix for newer keras
dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
model(dummy)

# ============================================================
# LOAD CIFAR10
# ============================================================

(_, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype("float32") / 255.0

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ============================================================
# PICK IMAGE
# ============================================================

IMAGE_INDEX = 0

image = x_test[IMAGE_INDEX]

label = class_names[y_test[IMAGE_INDEX][0]]

input_image = np.expand_dims(image, axis=0)

# ============================================================
# SAVE ORIGINAL IMAGE
# ============================================================

plt.figure(figsize=(5,5))

plt.imshow(image)

plt.title(f"Original Image\nTrue Label: {label}")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_original_image.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Saved: 01_original_image.png")

# ============================================================
# BUILD ACTIVATION MODEL
# ============================================================

conv_layers = []

for layer in model.layers:

    if isinstance(layer, tf.keras.layers.Conv2D):

        conv_layers.append(layer)

activation_model = Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in conv_layers]
)

activations = activation_model.predict(input_image)

# ============================================================
# VISUALIZE FEATURE MAPS
# ============================================================

for layer_index, activation in enumerate(activations):

    layer_name = conv_layers[layer_index].name

    print(f"\nProcessing {layer_name}")

    # --------------------------------------------------------
    # FEATURE MAP GRID
    # --------------------------------------------------------

    fig, axes = plt.subplots(3, 3, figsize=(10,10))

    fig.suptitle(
        f"{layer_name} Feature Maps\n"
        f"Different learned visual detectors",
        fontsize=18
    )

    for i, ax in enumerate(axes.flat):

        fmap = activation[0, :, :, i]

        ax.imshow(fmap, cmap='viridis')

        ax.set_title(f"Feature Map {i}")

        ax.axis('off')

    plt.tight_layout()

    filename = f"02_{layer_name}_feature_maps.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved: {filename}")

    # --------------------------------------------------------
    # MEAN ACTIVATION MAP
    # --------------------------------------------------------

    mean_map = np.mean(activation[0], axis=-1)

    plt.figure(figsize=(7,7))

    plt.imshow(mean_map, cmap='inferno')

    plt.title(
        f"{layer_name} Mean Activation\n"
        f"Bright = Important Regions"
    )

    plt.colorbar()

    plt.axis('off')

    filename = f"03_{layer_name}_mean_activation.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved: {filename}")

    # --------------------------------------------------------
    # ACTIVATION DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(8,4))

    plt.hist(activation.flatten(), bins=50)

    plt.title(
        f"{layer_name} Activation Distribution"
    )

    plt.xlabel("Activation Value")

    plt.ylabel("Count")

    filename = f"04_{layer_name}_activation_histogram.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved: {filename}")

# ============================================================
# VISUALIZE LEARNED FILTERS
# ============================================================

for idx, layer in enumerate(conv_layers):

    filters = layer.get_weights()[0]

    fig, axes = plt.subplots(3, 3, figsize=(8,8))

    fig.suptitle(
        f"{layer.name} Learned Filters\n"
        f"Pattern Detectors Learned During Training",
        fontsize=18
    )

    for i, ax in enumerate(axes.flat):

        filt = filters[:, :, 0, i]

        ax.imshow(filt, cmap='gray')

        ax.set_title(f"Kernel {i}")

        ax.axis('off')

    plt.tight_layout()

    filename = f"05_{layer.name}_filters.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved: {filename}")

# ============================================================
# DENSE LAYER ACTIVATIONS
# ============================================================

dense_layers = []

for layer in model.layers:

    if isinstance(layer, tf.keras.layers.Dense):

        dense_layers.append(layer)

dense_model = Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in dense_layers]
)

dense_outputs = dense_model.predict(input_image)

for idx, dense_output in enumerate(dense_outputs):

    activ = dense_output[0]

    padded = np.zeros(64)

    usable = min(64, len(activ))

    padded[:usable] = activ[:usable]

    heatmap = padded.reshape(8,8)

    plt.figure(figsize=(7,7))

    plt.imshow(heatmap, cmap='hot')

    plt.title(
        f"Dense Layer {idx+1}\n"
        f"Neuron Activation Heatmap"
    )

    plt.colorbar()

    plt.axis('off')

    filename = f"06_dense_layer_{idx+1}.png"

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved: {filename}")

# ============================================================
# FINAL PREDICTION
# ============================================================

prediction = model.predict(input_image)

probabilities = tf.nn.softmax(prediction[0]).numpy()

predicted_class = np.argmax(probabilities)

plt.figure(figsize=(10,4))

bars = plt.bar(class_names, probabilities)

bars[predicted_class].set_color('red')

plt.title(
    f"Final Prediction\n"
    f"Predicted: {class_names[predicted_class]}"
)

plt.ylabel("Confidence")

plt.xticks(rotation=45)

filename = "07_prediction_probabilities.png"

plt.savefig(
    os.path.join(OUTPUT_DIR, filename),
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Saved: {filename}")

# ============================================================
# NETWORK SHAPE FLOW
# ============================================================

shape_file = os.path.join(
    OUTPUT_DIR,
    "08_shape_evolution.txt"
)

with open(shape_file, "w") as f:

    f.write("CNN SHAPE EVOLUTION\n")
    f.write("===========================\n\n")

    f.write(f"Input Shape: {input_image.shape}\n\n")

    for layer in model.layers:

        temp_model = Model(
            inputs=model.inputs,
            outputs=layer.output
        )

        output = temp_model.predict(input_image)

        line = f"{layer.name:20s} -> {output.shape}\n"

        print(line.strip())

        f.write(line)

print("Saved: 08_shape_evolution.txt")

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n================================================")
print("ALL VISUALIZATIONS SAVED")
print("================================================")

print(f"\nSaved inside folder:\n{OUTPUT_DIR}\n")