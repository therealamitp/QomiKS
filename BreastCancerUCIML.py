import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap       # quantum feature map
from qiskit.primitives import StatevectorSampler       # V2 sampler (replacement for Sampler)
from qiskit_machine_learning.kernels import FidelityQuantumKernel
#referenced data: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# 1. Load data and split into train/test
X, y = load_breast_cancer(return_X_y=True)


X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# 2. Select two features and map to angles in [0, π]
def normalize_to_pi(arr):
    min_, max_ = arr.min(axis=0), arr.max(axis=0)
    return (arr - min_) / (max_ - min_) * np.pi

X_train = normalize_to_pi(X_train_raw[:, [0, 1]])
X_test  = normalize_to_pi(X_test_raw[:,  [0, 1]])

# ——— NEW: Classical Gram matrix (inner-product) ———
G_train = X_train @ X_train.T      # shape (n_train, n_train)
G_test  = X_test  @ X_train.T      # shape (n_test,  n_train)
print("Classical Gram matrix (train):\n", G_train)
print("Classical Gram matrix (test):\n",  G_test)

# 3. Build quantum states via angle-encoding and Kronecker product
def quantum_state(thetas):
    θ1, θ2 = thetas
    q0 = np.array([np.cos(θ1/2), np.sin(θ1/2)])
    q1 = np.array([np.cos(θ2/2), np.sin(θ2/2)])
    return np.kron(q0, q1)

states_train = np.array([quantum_state(t) for t in X_train])
states_test  = np.array([quantum_state(t) for t in X_test])

# 4. Compute quantum kernel: K_ij = |⟨ψ_i|ψ_j⟩|²
K_train = np.abs(states_train @ states_train.conj().T) ** 2
K_test  = np.abs(states_test  @ states_train.conj().T) ** 2

# 5. Train classical SVM with precomputed quantum kernel
svc_q = SVC(kernel='precomputed')
svc_q.fit(K_train, y_train)
acc_q = svc_q.score(K_test, y_test)
print(f"Quantum-kernel SVM accuracy: {acc_q:.3f}")

# 6. (Optional) Train classical SVM on plain Gram matrix to compare
svc_c = SVC(kernel='precomputed')
svc_c.fit(G_train, y_train)
acc_c = svc_c.score(G_test, y_test)
print(f"Classical-Gram SVM accuracy: {acc_c:.3f}")

# 7. Qiskit feature-map sanity check (negligible overhead)
feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
print(feature_map.decompose().draw(output='text'))

# 8. Instantiate the V2 sampler (not actually used above, but available)
sampler = StatevectorSampler()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

import matplotlib.pyplot as plt
import numpy as np

# === Step 6: Visualize the quantum kernel matrix with diagonal ===
K = K_train  # or whatever your kernel matrix is

plt.figure(figsize=(6,6))
plt.imshow(
    K,
    cmap='viridis',
    interpolation='nearest',
    vmin=0.0, vmax=1.0  # optional: stretch full 0→1 scale
)
plt.colorbar(label='Kernel value')

# overlay a contrasting diagonal
n = K.shape[0]
plt.plot(
    np.arange(n),      # x-coordinates
    np.arange(n),      # y-coordinates
    color='yellow',       # pick something that stands out
    linewidth=2,
)

plt.title("Quantum Kernel Gram Matrix")
plt.xlabel("Sample index")
plt.ylabel("Sample index")
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.show()
