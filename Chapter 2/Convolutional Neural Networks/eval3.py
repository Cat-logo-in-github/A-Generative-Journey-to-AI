import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import cifar10

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "CNN_Interpretability"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(
    "cifar10_cnn_model.keras",
    compile=False
)

# Force-build model graph
_ = model(tf.zeros((1, 32, 32, 3)))

# ============================================================
# LOAD CIFAR10
# ============================================================

(_, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype("float32") / 255.0

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# ============================================================
# SELECT IMAGE
# ============================================================

IMAGE_INDEX = 0

image = x_test[IMAGE_INDEX]

input_image = np.expand_dims(image, axis=0)

true_label = class_names[y_test[IMAGE_INDEX][0]]

# ============================================================
# SAVE ORIGINAL IMAGE
# ============================================================

plt.figure(figsize=(5,5))

plt.imshow(image)

plt.title(f"Original Image\nTrue Label: {true_label}")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_original_image.png"),
    dpi=300
)

plt.close()

print("Saved: 01_original_image.png")

# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(input_image, verbose=0)

probabilities = tf.nn.softmax(prediction[0]).numpy()

predicted_index = np.argmax(probabilities)

predicted_class = class_names[predicted_index]

# ============================================================
# SAVE PREDICTION BAR CHART
# ============================================================

plt.figure(figsize=(10,4))

bars = plt.bar(class_names, probabilities)

bars[predicted_index].set_color("red")

plt.title(
    f"Prediction Confidence\n"
    f"Predicted: {predicted_class}"
)

plt.ylabel("Confidence")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_prediction_probabilities.png"
    ),
    dpi=300
)

plt.close()

print("Saved: 02_prediction_probabilities.png")

# ============================================================
# GET ALL CONV LAYERS
# ============================================================

conv_layers = []

for layer in model.layers:

    if isinstance(layer, tf.keras.layers.Conv2D):

        conv_layers.append(layer)

print("\nConvolution Layers:")

for layer in conv_layers:

    print(layer.name)

# ============================================================
# CREATE ACTIVATION MODEL
# ============================================================

activation_model = Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in conv_layers]
)

# ============================================================
# FORWARD PASS THROUGH CNN
# ============================================================

feature_maps = activation_model.predict(
    input_image,
    verbose=0
)

# ============================================================
# VISUALIZE FEATURE MAPS OF EACH LAYER
# ============================================================

for layer_idx, fmap in enumerate(feature_maps):

    layer_name = conv_layers[layer_idx].name

    fmap = fmap[0]

    n_features = fmap.shape[-1]

    n_show = min(16, n_features)

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(10,10)
    )

    fig.suptitle(
        f"Feature Maps - {layer_name}",
        fontsize=18
    )

    for i, ax in enumerate(axes.flat):

        if i < n_show:

            ax.imshow(
                fmap[:, :, i],
                cmap='viridis'
            )

            ax.set_title(f"Map {i}")

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"03_featuremaps_{layer_name}.png"
        ),
        dpi=300
    )

    plt.close()

    print(f"Saved feature maps: {layer_name}")

# ============================================================
# VISUALIZE LEARNED KERNELS
# ============================================================

for layer in conv_layers:

    filters = layer.get_weights()[0]

    num_filters = filters.shape[-1]

    n_show = min(16, num_filters)

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(10,10)
    )

    fig.suptitle(
        f"Learned Kernels - {layer.name}",
        fontsize=18
    )

    for i, ax in enumerate(axes.flat):

        if i < n_show:

            kernel = filters[:, :, 0, i]

            ax.imshow(
                kernel,
                cmap='gray'
            )

            ax.set_title(f"Kernel {i}")

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"04_kernels_{layer.name}.png"
        ),
        dpi=300
    )

    plt.close()

    print(f"Saved kernels: {layer.name}")

# ============================================================
# LAST CONV LAYER
# ============================================================

last_conv_layer = conv_layers[-1]

print("\nLast Conv Layer:", last_conv_layer.name)

# ============================================================
# CREATE SAFE GRAD-CAM MODEL
# ============================================================

inputs = tf.keras.Input(shape=(32,32,3))

x = inputs

last_conv_output = None

for layer in model.layers:

    x = layer(x)

    if layer.name == last_conv_layer.name:

        last_conv_output = x

grad_model = tf.keras.Model(
    inputs=inputs,
    outputs=[last_conv_output, x]
)

# ============================================================
# COMPUTE GRAD-CAM
# ============================================================

input_tensor = tf.convert_to_tensor(input_image)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_tensor,
        training=False
    )

    class_score = predictions[:, predicted_index]

grads = tape.gradient(
    class_score,
    conv_outputs
)

if grads is None:

    raise RuntimeError(
        "Gradients are None."
    )

# ============================================================
# CHANNEL IMPORTANCE
# ============================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0,1,2)
)

conv_outputs = conv_outputs[0].numpy()

pooled_grads = pooled_grads.numpy()

# ============================================================
# VISUALIZE CHANNEL IMPORTANCE
# ============================================================

plt.figure(figsize=(12,4))

plt.bar(
    np.arange(len(pooled_grads)),
    pooled_grads
)

plt.title(
    "Importance of Final Conv Channels"
)

plt.xlabel("Channel")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_channel_importance.png"
    ),
    dpi=300
)

plt.close()

print("Saved: 05_channel_importance.png")

# ============================================================
# TOP CHANNELS
# ============================================================

top_channels = np.argsort(
    pooled_grads
)[-9:]

# ============================================================
# VISUALIZE IMPORTANT FEATURE MAPS
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(10,10)
)

fig.suptitle(
    "Most Important Feature Maps",
    fontsize=18
)

for i, ax in enumerate(axes.flat):

    channel = top_channels[i]

    fmap = conv_outputs[:, :, channel]

    ax.imshow(
        fmap,
        cmap='jet'
    )

    ax.set_title(
        f"Channel {channel}\n"
        f"{pooled_grads[channel]:.3f}"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_top_feature_maps.png"
    ),
    dpi=300
)

plt.close()

print("Saved: 06_top_feature_maps.png")

# ============================================================
# VISUALIZE IMPORTANT KERNELS
# ============================================================

filters = last_conv_layer.get_weights()[0]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(10,10)
)

fig.suptitle(
    "Most Important Kernels",
    fontsize=18
)

for i, ax in enumerate(axes.flat):

    channel = top_channels[i]

    kernel = filters[:, :, 0, channel]

    ax.imshow(
        kernel,
        cmap='gray'
    )

    ax.set_title(
        f"Kernel {channel}"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_important_kernels.png"
    ),
    dpi=300
)

plt.close()

print("Saved: 07_important_kernels.png")

# ============================================================
# COMPUTE HEATMAP
# ============================================================

heatmap = np.zeros(
    conv_outputs.shape[:2],
    dtype=np.float32
)

for i in range(len(pooled_grads)):

    heatmap += (
        pooled_grads[i]
        * conv_outputs[:, :, i]
    )

heatmap = np.maximum(
    heatmap,
    0
)

heatmap /= (
    np.max(heatmap) + 1e-8
)

# ============================================================
# SAVE RAW HEATMAP
# ============================================================

plt.figure(figsize=(6,6))

plt.imshow(
    heatmap,
    cmap='jet'
)

plt.title("Grad-CAM Heatmap")

plt.colorbar()

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_gradcam_heatmap.png"
    ),
    dpi=300
)

plt.close()

print("Saved: 08_gradcam_heatmap.png")

# ============================================================
# OVERLAY
# ============================================================

heatmap_resized = cv2.resize(
    heatmap,
    (32,32)
)

heatmap_uint8 = np.uint8(
    255 * heatmap_resized
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

original_uint8 = np.uint8(
    image * 255
)

overlay = cv2.addWeighted(
    original_uint8,
    0.6,
    heatmap_color,
    0.4,
    0
)

overlay = cv2.cvtColor(
    overlay,
    cv2.COLOR_BGR2RGB
)

# ============================================================
# SAVE OVERLAY
# ============================================================

plt.figure(figsize=(6,6))

plt.imshow(overlay)

plt.title(
    f"Grad-CAM Overlay\n"
    f"Prediction: {predicted_class}"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "09_gradcam_overlay.png"
    ),
    dpi=300
)

plt.close()

print("Saved: 09_gradcam_overlay.png")

# ============================================================
# COMPLETE PIPELINE VISUAL
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18,5)
)

axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(
    conv_outputs[:, :, top_channels[-1]],
    cmap='viridis'
)
axes[1].set_title("Top Feature Map")
axes[1].axis("off")

axes[2].imshow(
    heatmap,
    cmap='jet'
)
axes[2].set_title("Grad-CAM")
axes[2].axis("off")

axes[3].imshow(overlay)
axes[3].set_title("Overlay")
axes[3].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "10_full_pipeline.png"
    ),
    dpi=300
)

plt.close()

print("Saved: 10_full_pipeline.png")

# ============================================================
# EXPLANATION FILE
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "11_explanation.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CNN INTERPRETABILITY REPORT\n"
    )

    f.write(
        "===============================\n\n"
    )

    f.write(
        f"True Label: {true_label}\n"
    )

    f.write(
        f"Predicted Label: {predicted_class}\n\n"
    )

    f.write(
        "WHAT HAPPENS INSIDE CNN:\n\n"
    )

    f.write(
        "1. Early convolution layers detect:\n"
    )

    f.write(
        "   - edges\n"
        "   - textures\n"
        "   - orientations\n\n"
    )

    f.write(
        "2. Middle layers combine patterns into:\n"
    )

    f.write(
        "   - shapes\n"
        "   - contours\n"
        "   - object parts\n\n"
    )

    f.write(
        "3. Final convolution layers detect:\n"
    )

    f.write(
        "   - class-specific structures\n"
        "   - semantic object regions\n\n"
    )

    f.write(
        "4. Grad-CAM backtracks importance from:\n"
    )

    f.write(
        "   prediction → gradients → feature maps\n"
    )

    f.write(
        "5. Important kernels are the filters that\n"
    )

    f.write(
        "   generated influential activations.\n"
    )

print("Saved: 11_explanation.txt")

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n=====================================")
print("CNN INTERPRETABILITY COMPLETE")
print("=====================================")

print(f"\nSaved inside:\n{OUTPUT_DIR}")