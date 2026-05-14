import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import cv2

# =========================
# FOLDERS
# =========================
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)

# =========================
# LOAD DATA (WITH PROGRESS)
# =========================
photo_dir = r"D:\Desktop\Python test project\tflowplay\car-pics"
cache_file = "outputs/checkpoints/dataset.npy"

if os.path.exists(cache_file):
    print("Loading cached dataset...")
    car_images = np.load(cache_file)

else:
    print("Loading dataset from images...")

    car_images = []
    files = os.listdir(photo_dir)

    valid = 0
    skipped = 0

    for i, file in enumerate(files):

        if i % 500 == 0:
            print(f"Processed {i}/{len(files)} images")

        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(photo_dir, file)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                skipped += 1
                continue

            try:
                img = cv2.resize(img, (28, 28))
                img = img.astype("float32")
                img = (img - 127.5) / 127.5

                car_images.append(img)
                valid += 1

            except:
                skipped += 1

    car_images = np.array(car_images)
    car_images = np.expand_dims(car_images, axis=-1)

    np.save(cache_file, car_images)

    print(f"Dataset loaded. Valid: {valid}, Skipped: {skipped}")

print("Dataset shape:", car_images.shape)

# =========================
# GENERATOR
# =========================
generator = keras.Sequential([
    keras.Input(shape=(100,)),

    layers.Dense(7 * 7 * 256),
    layers.Reshape((7, 7, 256)),

    layers.Conv2DTranspose(128, 5, strides=1, padding="same"),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),

    layers.Conv2DTranspose(64, 5, strides=2, padding="same"),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),

    layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh"),
])

# =========================
# DISCRIMINATOR
# =========================
discriminator = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    layers.Conv2D(64, 5, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),

    layers.Conv2D(128, 5, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(1, activation="sigmoid"),
])

discriminator.compile(
    optimizer=keras.optimizers.Adam(0.0001, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# GAN
# =========================
discriminator.trainable = False

gan_input = keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = keras.Model(gan_input, gan_output)

gan.compile(
    optimizer=keras.optimizers.Adam(0.0001, beta_1=0.5),
    loss="binary_crossentropy"
)

# =========================
# FIXED NOISE (TEACHING VISUALIZATION)
# =========================
fixed_noise = np.random.normal(0, 1, (10, 100))

def save_images(epoch, model, folder):
    gen = model.predict(fixed_noise, verbose=0)
    gen = (gen + 1) / 2.0

    fig, axs = plt.subplots(1, 10, figsize=(15, 2))

    for i in range(10):
        axs[i].imshow(gen[i].reshape(28, 28), cmap="gray")
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{folder}/epoch_{epoch}.png")
    plt.close()

# =========================
# TRAINING
# =========================
epochs = 17000
batch_size = 32

d_losses, g_losses = [], []

print("Training started...")

for epoch in range(epochs):

    # -------------------------
    # TRAIN DISCRIMINATOR
    # -------------------------
    discriminator.trainable = True

    idx = np.random.randint(0, car_images.shape[0], batch_size)
    real = car_images[idx]

    noise = np.random.normal(0, 1, (batch_size, 100))
    fake = generator.predict(noise, verbose=0)

    real_y = np.ones((batch_size, 1)) * 0.9
    fake_y = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real, real_y)
    d_loss_fake = discriminator.train_on_batch(fake, fake_y)

    d_loss = 0.5 * (np.array(d_loss_real) + np.array(d_loss_fake))

    # -------------------------
    # TRAIN GENERATOR
    # -------------------------
    discriminator.trainable = False

    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    g_loss = float(g_loss)

    # -------------------------
    # LOGGING
    # -------------------------
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)

    if epoch % 100 == 0:
        print(f"{epoch} | D: {d_loss[0]:.4f} | G: {g_loss:.4f}")

    # -------------------------
    # CHECKPOINTS (IMPORTANT FIX)
    # -------------------------

    if epoch % 1000 == 0 and epoch != 0:

        print("Saving FULL checkpoint...")

        save_images(epoch, generator, "outputs/images")

        # Save BOTH models
        generator.save(f"outputs/models/gen_{epoch}.keras")
        discriminator.save(f"outputs/models/disc_{epoch}.keras")

# =========================
# FINAL PLOTS
# =========================
plt.plot(d_losses, label="D loss")
plt.plot(g_losses, label="G loss")
plt.legend()
plt.savefig("outputs/images/loss_curve.png")
plt.show()

print("Training complete.")