import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import cv2

# -------------------------
# FOLDERS
# -------------------------
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

# -------------------------
# LOAD DATA SAFELY + FAST
# -------------------------
photo_dir = r"D:\Desktop\Python test project\tflowplay\car-pics"

cache_file = "outputs/checkpoints/dataset.npy"

# Load cached dataset if exists (VERY IMPORTANT)
if os.path.exists(cache_file):
    print("Loading cached dataset...")
    car_images = np.load(cache_file)

else:
    print("Loading images from disk...")

    car_images = []
    files = os.listdir(photo_dir)

    for i, filename in enumerate(files):

        if i % 50 == 0:
            print(f"Loading image {i}/{len(files)}")

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(photo_dir, filename)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("Skipping broken image:", filename)
                continue

            try:
                img = cv2.resize(img, (28, 28))
                img = img.astype("float32") / 255.0
                car_images.append(img)
            except:
                print("Error processing:", filename)
                continue

    car_images = np.array(car_images)
    car_images = np.expand_dims(car_images, axis=-1)

    np.save(cache_file, car_images)
    print("Dataset saved to cache!")

print("Dataset shape:", car_images.shape)

# -------------------------
# GENERATOR
# -------------------------
generator = keras.Sequential([
    keras.Input(shape=(100,)),
    layers.Dense(7 * 7 * 256),
    layers.Reshape((7, 7, 256)),

    layers.Conv2DTranspose(128, 5, strides=1, padding="same"),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.2),

    layers.Conv2DTranspose(64, 5, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.2),

    layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="sigmoid"),
])

# -------------------------
# DISCRIMINATOR
# -------------------------
discriminator = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    layers.Conv2D(64, 5, strides=2, padding="same"),
    layers.LeakyReLU(negative_slope=0.2),

    layers.Conv2D(128, 5, strides=2, padding="same"),
    layers.LeakyReLU(negative_slope=0.2),

    layers.Flatten(),
    layers.Dense(1, activation="sigmoid"),
])

discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0002, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# GAN
# -------------------------
discriminator.trainable = False

gan_input = keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = keras.Model(gan_input, gan_output)
gan.compile(
    optimizer=keras.optimizers.Adam(0.0002, beta_1=0.5),
    loss="binary_crossentropy"
)

# -------------------------
# IMAGE SAVER
# -------------------------
def save_generated_images(epoch, generator):
    noise = np.random.normal(0, 1, (10, 100))
    gen = generator.predict(noise, verbose=0)
    gen = gen.reshape(10, 28, 28)

    fig, axs = plt.subplots(1, 10, figsize=(15, 2))

    for i in range(10):
        axs[i].imshow(gen[i], cmap="gray")
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/images/generated_{epoch}.png")
    plt.close()

# -------------------------
# TRAINING SETTINGS
# -------------------------
batch_size = 32
epochs = 500

d_losses, g_losses, d_accs = [], [], []

# -------------------------
# TRAIN LOOP
# -------------------------
print("Starting training...")

for epoch in range(epochs):

    idx = np.random.randint(0, car_images.shape[0], batch_size)
    real_images = car_images[idx]

    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator(noise, training=False).numpy()

    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))

    d_loss = 0.5 * (np.array(d_loss_real) + np.array(d_loss_fake))

    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    g_loss = float(np.array(g_loss).mean())

    d_losses.append(d_loss[0])
    d_accs.append(d_loss[1])
    g_losses.append(g_loss)

    if epoch % 100 == 0:
        print(f"{epoch} | D: {d_loss[0]:.4f} | Acc: {d_loss[1]*100:.2f}% | G: {g_loss:.4f}")

    # -------------------------
    # CHECKPOINTS
    # -------------------------
    if epoch % 1000 == 0 and epoch != 0:
        print("Saving checkpoint...")

        save_generated_images(epoch, generator)

        generator.save(f"outputs/models/generator_{epoch}.keras")
        discriminator.save(f"outputs/models/discriminator_{epoch}.keras")

# -------------------------
# Final training plots
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.legend()
plt.title("GAN Training Loss")
plt.savefig("outputs/images/loss_plot.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(d_accs, label="Discriminator Accuracy")
plt.legend()
plt.title("Discriminator Accuracy")
plt.savefig("outputs/images/accuracy_plot.png")
plt.show()


print("Training complete!")