# 🧠 Unsupervised Learning --> PCA (Revision Notes)

## 📌 What is Unsupervised Learning?

Unsupervised learning is a type of machine learning where:

* ❌ No labels are given
* ✔ Only input data `X` is used
* ✔ The model finds patterns on its own

### Example:

```text id="u1"
Images of digits (no answers provided)
→ model finds structure in data
```

---

## 🔑 Key Idea

> The model is NOT learning “correct answers”
> It is learning “patterns in the data”

---

## 📊 Types of Unsupervised Learning

### 1. Clustering

Groups similar data together
Examples:

* K-Means
* Hierarchical clustering

### 2. Dimensionality Reduction

Reduces number of features while keeping structure
Examples:

* PCA (Principal Component Analysis)
* t-SNE
* UMAP

---

# 📉 PCA (Principal Component Analysis)

## 📌 What is PCA?

PCA is a **dimensionality reduction technique** that:

> Converts high-dimensional data into lower dimensions while preserving maximum variance.

---

## 🧠 Simple Intuition

Imagine:

* Data exists in 64 dimensions (digits dataset)
* We cannot visualize 64D
* PCA compresses it into 2D or 3D for visualization

---

## ⚙️ What PCA Actually Does

PCA:

1. Finds directions where data varies most
2. Creates new axes called **Principal Components**
3. Projects data onto these axes

---

## 📐 Principal Components

* **PC1 (First Principal Component)**
  Direction of maximum variance

* **PC2 (Second Principal Component)**
  Second most important direction (perpendicular to PC1)

---

## 📉 Important Property

✔ PCs are combinations of ALL original features
❌ They are NOT original features themselves

---

## 🧾 Example Flow

```text id="u2"
64D digit images
↓
PCA
↓
2D representation
↓
Plot visualization
```

---

# 🖼️ What the PCA Plot Shows

Each dot represents:

* One handwritten digit image

Color represents:

* True digit label (0–9)

---

## 📌 Why clusters appear

Digits with similar shapes:

* look similar in pixel space
* get mapped near each other

Example:

* many “1”s cluster together
* many “0”s cluster together

---

## ⚠️ Why overlap happens

When reducing dimensions:

```text id="u3"
64D → 2D
```

Information is lost.

So:

* different digits may overlap
* multiple points can share similar coordinates

👉 This is expected, not an error.

---

## ❗ Key Misconception

PCA is NOT trying to:

* separate classes
* classify digits
* avoid overlap

It IS trying to:

* preserve variance
* compress data efficiently

---

# 🎯 What PCA is Good For

✔ Data visualization
✔ Feature compression
✔ Noise reduction
✔ Preprocessing before ML models

---

# 🚫 What PCA is NOT Good For

❌ Classification
❌ Perfect separation of classes
❌ Understanding labels

---

# 🧪 Code Summary

```python id="u4"
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.show()
```

---

# 🧠 Key Takeaways

* PCA reduces dimensions, not improves classification
* It is unsupervised (no labels used in training)
* Overlap in plots is normal due to information loss
* It helps visualize high-dimensional data in 2D/3D
* It finds directions of maximum variance, not class boundaries

---

### NOTE: It’s **arbitrary in PCA space** — PCA does *not* label anything.

So the honest answer is:

> You do NOT know a cluster is “0” from PCA alone.
> You infer it by comparing with the known labels (`y`).

---

# 🧠 What is actually happening

In your plot:

* PCA creates a 2D/3D coordinate system
* Each point is placed based on similarity of pixel patterns
* You color points using the real labels (`y`)

So:

```text id="a1"
Position (PC1, PC2, PC3) → from PCA
Color (0–9) → from dataset labels
```

---

# 📌 Important correction

When you see a yellow cluster:

That does NOT mean:

❌ “PCA discovered digit 0”

It means:

✔ “Many points labeled 0 happen to fall in this region”

---

# 🧠 So is the cluster identity known?

Only because YOU already have labels.

PCA itself has no idea.

It only knows:

> “these points are similar in feature space”

---

# 🔥 If labels were removed completely

If you did:

```python id="a2"
y = None
```

and plotted only PCA:

* you would still see clusters
* but you would NOT know what digit each cluster represents

You’d just see “groups of similar shapes”.

---

# 📊 So how do we “map” clusters to digits?

We do this AFTER PCA:

### Step 1: look at cluster region

### Step 2: check majority label inside that region

Example:

```text id="a3"
Cluster A → 85% of points are digit 0
Cluster B → mostly digit 1
```

Then we *interpret*:

* Cluster A ≈ “0”
* Cluster B ≈ “1”

But this is not PCA doing it — it’s post-analysis.

---

# 🧠 Key idea: PCA ≠ classifier

| Task                 | PCA does it? |
| -------------------- | ------------ |
| Find structure       | ✔ yes        |
| Group similar points | ✔ indirectly |
| Assign meaning (0–9) | ❌ no         |
| Understand digits    | ❌ no         |

---

# 🚨 Very important insight

PCA space is like:

> a map without names

You see:

* neighborhoods (clusters)
* distances
* structure

But:

* no labels
* no semantic meaning

You add meaning afterward.

---

# 🧩 Analogy

Imagine you sorted books only by:

* thickness
* paper density
* ink distribution

You might get clusters like:

* novels
* textbooks
* comics

But the algorithm doesn’t know “novel”.

You just observe:

> “this cluster mostly contains novels”

---

# 🧠 So your question answered precisely:

> “How do we know yellow cluster is 0?”

✔ You don’t know from PCA
✔ You infer it using labels after the fact
✔ It is NOT inherently labeled by PCA
✔ The cluster is meaningful, but not named

---

# 🔥 Final takeaway

PCA gives you:

> structure without meaning

You provide:

> meaning using labels or interpretation

---

