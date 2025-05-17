import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
#referenced data: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
# === 1. Load features and labels from separate CSVs ===
data_dir = "/Users/amitprakash/Desktop/School/Projects/Quantum Cancer MSI/TCGA-PANCAN-HiSeq-801x20531"
X_df = pd.read_csv(f"{data_dir}/data.csv", index_col=0)    # assumes first column is sample ID
y_df = pd.read_csv(f"{data_dir}/labels.csv", index_col=0)  # same sample ID index

# align by index just in case
X_df = X_df.loc[y_df.index]

X = X_df.values      # shape (801, 20531)
y = y_df.values.flatten()

# === 2. Split into train/test ===
X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.9, stratify=y, random_state=42
)

# === 3. Reduce to 2 dimensions via PCA ===
pca = PCA(n_components=2, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_raw)
X_te_pca = pca.transform(X_te_raw)

# === 4. Normalize each axis into [0, π] ===
def normalize_to_pi(arr):
    mn, mx = arr.min(axis=0), arr.max(axis=0)
    return (arr - mn) / (mx - mn) * np.pi

X_train = normalize_to_pi(X_tr_pca)
X_test  = normalize_to_pi(X_te_pca)

# === 5. Classical Gram matrices ===
G_train = X_train @ X_train.T
G_test  = X_test  @ X_train.T

# === 6. Quantum state encoding (angle → 2-qubit state) ===
def quantum_state(thetas):
    θ1, θ2 = thetas
    q0 = np.array([np.cos(θ1/2), np.sin(θ1/2)])
    q1 = np.array([np.cos(θ2/2), np.sin(θ2/2)])
    return np.kron(q0, q1)

states_train = np.array([quantum_state(t) for t in X_train])
states_test  = np.array([quantum_state(t) for t in X_test])

# === 7. Compute quantum kernels: K = |⟨ψ_i|ψ_j⟩|² ===
K_train = np.abs(states_train @ states_train.conj().T) ** 2
K_test  = np.abs(states_test  @ states_train.conj().T) ** 2

# === 8. Train & evaluate Quantum-kernel SVM ===
svc_q = SVC(kernel='precomputed')
svc_q.fit(K_train, y_train)
print(f"Quantum-kernel SVM accuracy: {svc_q.score(K_test, y_test):.3f}")

# === 9. Train & evaluate Classical-Gram SVM ===
svc_c = SVC(kernel='precomputed')
svc_c.fit(G_train, y_train)
print(f"Classical-Gram SVM accuracy: {svc_c.score(G_test, y_test):.3f}")

# === 10. Sanity-check feature map circuit (no heavy sim) ===
feature_map = ZZFeatureMap(feature_dimension=4, reps=3)
print(feature_map.decompose().draw(output='text'))

# === 11. (Optional) Instantiate Qiskit kernel interface ===
sampler = StatevectorSampler()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# === 12. Visualize the quantum kernel matrix ===
plt.figure(figsize=(6,6))
plt.imshow(K_train, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
plt.colorbar(label='Kernel value')
n = K_train.shape[0]
plt.plot(np.arange(n), np.arange(n), color='yellow', linewidth=2)
plt.title("Quantum Kernel Gram Matrix")
plt.xlabel("Sample index")
plt.ylabel("Sample index")
plt.tight_layout()
plt.show()
