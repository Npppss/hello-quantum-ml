import numpy as np
import matplotlib.pyplot as plt

# Komponen Qiskit Machine Learning
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

##Scikit-learn untuk Dataset dan Data Splitting
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler

# Persiapkan Dataset
print("Persiapan dataset klasik...")
# Membuat dataset dummy: 100 titik data, 2 fitur (X dan Y), 2 kelas (0 dan 1)
X, y=make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

# Normalisasi data (penting untuk rotasi sudut kuantum)
# Kita skala data antara 0 dan 1, atau -pi hingga pi
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Qiskit VQC mengharapkan label dalam format One-Hot Encoding
# Kelas 0 -> [1, 0] | Kelas 1 -> [0, 1]
y_onehot = np.zeros((y.size, y.max() + 1))
y_onehot[np.arange(y.size), y] = 1

# Membagi data menjadi 80% Training dan 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Arsitektur Kuantum (QPU)
