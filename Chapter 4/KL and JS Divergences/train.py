import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
This Code is for showing Kullback-Leibler divergence and Jensen-Shannon divergence between two 
probability distributions. It is not directly related to the training of a model, but it can 
be useful for understanding the concepts of divergence in the context of machine learning 
and statistics.
"""

# Generate two probability distributions (e.g., normal distributions)
x = np.linspace(-5, 5, 1000)
pdf1 = norm.pdf(x, loc=0, scale=1)  # First distribution (standard normal)
pdf2 = norm.pdf(x, loc=1, scale=1)  # Second distribution (shifted and normal)

# Plot the PDFs

plt.figure(figsize=(8, 6))
plt.plot(x, pdf1, label='Distribution 1 (N(0,1))')
plt.plot(x, pdf2, label='Distribution 2 (N(1,1))')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Generated Probability Density Functions')
plt.legend()
plt.grid(True)

def kld(p, q):
    return np.sum(p * np.log(p / q))

def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)


# Calculate KL Divergence and JS Divergence

kl_divergence = kld(pdf1, pdf2)
js_divergence = jsd(pdf1, pdf2)

print(f"Kullback-Leibler Divergence (KL Divergence): {kl_divergence:.4f}")
print(f"Jensen-Shannon Divergence (JS Divergence): {js_divergence:.4f}")

# Plot the divergences as bar charts
plt.figure(figsize=(8, 6))
plt.bar(['KL Divergence', 'JS Divergence'], [kl_divergence, js_divergence])
plt.ylabel('Divergence Value')
plt.title('Comparison of Divergences')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()