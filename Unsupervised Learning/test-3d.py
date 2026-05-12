import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Apply PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Extract components explicitly
pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]
pc3 = X_pca[:, 2]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    pc1, pc2, pc3,
    c=y,
    cmap='viridis'
)

# Axis labels (explicit meaning)
ax.set_xlabel('Principal Component 1 (PC1)')
ax.set_ylabel('Principal Component 2 (PC2)')
ax.set_zlabel('Principal Component 3 (PC3)')
ax.set_title('PCA of Digits Dataset (3D)')

# Color legend
plt.colorbar(scatter, label='Digit Class', pad=0.1)

# PCA Component Heatmaps (To Extract Meaning)

pc1 = pca.components_[0].reshape(8, 8)
pc2 = pca.components_[1].reshape(8, 8)
pc3 = pca.components_[2].reshape(8, 8)

fig2, axes = plt.subplots(1, 3, figsize=(10, 3))

im = axes[0].imshow(pc1, cmap='coolwarm')
axes[0].set_title("PC1 (pattern)")
axes[0].axis("off")

axes[1].imshow(pc2, cmap='coolwarm')
axes[1].set_title("PC2 (pattern)")
axes[1].axis("off")

axes[2].imshow(pc3, cmap='coolwarm')
axes[2].set_title("PC3 (pattern)")
axes[2].axis("off")

plt.tight_layout()

cbar = fig2.colorbar(im, ax=axes, fraction=0.07, pad=0.02)

plt.show()