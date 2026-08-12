import numpy as np
import matplotlib.pyplot as plt

# Komponen Qiskit Machine Learning
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler # Menggunakan Sampler V2
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA


##Scikit-learn untuk Dataset dan Data Splitting
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
import warnings

# Mengabaikan warning deprecation agar terminal lebih bersih
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
print("Membangun arsitektur kuantum...")
num_features = X.shape[1] # 2 Fitur -> Kita butuh 2 Qubit

# A. Data Encoding (Feature Map)
# ZZFeatureMap baik untuk mengikat data ke dalam entanglement
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')

# B. Parameterized Quantum Circuit (Ansatz)
# RealAmplitudes menggunakan gerbang rotasi Ry dan CNOT
ansatz = RealAmplitudes(num_qubits=num_features, reps=2, entanglement='linear')

# Tampilkan struktur sirkuit ke terminal
print("\nBentuk Feature Map (Encoding):")
print(feature_map.decompose().draw())

# Mengonfigurasi optimizer CPU
print("\nMengonfigurasi optimizer VQC...")
# Gunakan optimizer COBYLA dengan maksimal 100 iterasi
optimizer = COBYLA(maxiter=100)
# Inisialisasi mesin pengeksekusi kuantum (Primitive)
sampler = StatevectorSampler() # Menggunakan Sampler V2

# TITIK KRUSIAL: Berikan nilai awal acak agar COBYLA bisa mulai bergerak
initial_point = np.random.random(ansatz.num_parameters)

# Array untuk menyimpan nilai error selama training
objective_func_values = []
def callback_graph(weights, obj_func_eval):
    # Callback ini akan dipanggil setiap kali optimizer selesai mengevaluasi parameter
    objective_func_values.append(obj_func_eval)

# Inisialisasi mesin pengeksekusi kuantum (Primitive)

# Menyatukan semuanya ke dalam kelas VQC
vqc_model=VQC(
    sampler=sampler, # Gunakan Sampler untuk evaluasi sirkuit
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    initial_point=initial_point,
    callback=callback_graph # Simpan jejak error
)

# training and evaluasi model
print("Mulai Training (Ini mungkin memakan waktu beberapa detik/menit)...")
# Melatih model (Fit)
vqc_model.fit(X_train, y_train)

print("Training Selesai!")

# menguji akurasi model pada data testing
train_score = vqc_model.score(X_train, y_train)
test_score = vqc_model.score(X_test, y_test)

print(f"\nAkurasi Training : {train_score:.2f}")
print(f"Akurasi Testing  : {test_score:.2f}")